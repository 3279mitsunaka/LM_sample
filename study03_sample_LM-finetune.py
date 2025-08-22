#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, pipeline, set_seed)

#---- setting ----#
# 今回は小規模なファインチューニングコードであるため global 変数に直書きしていますが、
# 可能であれば yaml ファイルなどの外部ファイルに保存した方が管理が楽になります。

MODEL_NAME = "google/gemma-3-270m-it"       # 基盤モデル名です。デフォルトで hugging face に接続します
# MODEL_NAME = "/path/to/local-model"       # ローカルモデルを指定する場合は /.../model名 にします
CSV_PATH = "dataset/kansaiben/kansaiben.csv"
FINAL_DIR = "./gemma3-kansaiben-final"
MAX_LEN = 512   # 最大トークン長
EPOCHS = 3
BATCH = 2
LEARNING_RATE = 1e-5    # 学習率
WARMUP_RATIO = 0.03     # 訓練の初期段階で学習率を徐々に上げる割合
MAX_GRAD_NORM = 1.0     # 最大勾配ノルム

SEED = 9999
MIN_LEN = 4  # 4文字以下のデータを除外する

# 性能確認用の入力プロンプトリスト
test_text_list = ["こんにちは！","自己紹介をお願いします。","ありがとうございます。"]


#---- 参考 ----#
# 研究室環境の Docker から Ollama モデルを python API で呼び出す場合は以下のように記述します。
# ファインチューニングしたモデルを Ollama にエクスポートして利用したい場合に参考にしてください。
#
# [案1]
# from langchain_ollama import OllamaLLM
# model = OllamaLLM(model="llama3", base_url="http://host.docker.internal:62723")
# response = llm.invoke("ではここで一句")
#
# [案2]
# from openai import OpenAI
# client = OpenAI(base_url="http://host.docker.internal:62723/v1", api_key="ollama")  # api_key はダミーでよい
# response = client.chat.completions.create(model="llama3", messages=[{"role":"user","content":"ではここで一句"}],)
#


# model名からモデル本体とトークナイザを取得します
def load_model_tok():
    # token 取得
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # model 取得
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager")
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    return model, tok

# データセットの前処理をします
def build_dataset(tok):
    # データセットが存在しない場合は終了
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} が見つからない")

    # データセット形式が不正である場合は終了
    df = pd.read_csv(CSV_PATH).fillna("")
    if not {"instruction","output"}.issubset(df.columns):
        raise ValueError("CSVに instruction, output 列が必要")

    # Chat Template で 1サンプル文字列を作成
    texts = []
    for ins, out in zip(df["instruction"], df["output"]):
        msgs = [{"role":"user","content":str(ins)},
                {"role":"assistant","content":str(out)}]
        texts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False))
    raw = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        enc = tok(batch["text"], truncation=True, padding=False, max_length=MAX_LEN)
        enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
        return enc

    ds = raw.map(tokenize, batched=True, remove_columns=["text"])
    # 短すぎるサンプルを除外（損失0対策）
    ds = ds.filter(lambda r: len(r["input_ids"]) >= MIN_LEN)
    return ds

# トークン長の異なる入力群を同じ長さにそろえてテンソルのバッチにする。 DataLoader は長さの異なるデータを扱えない。
def data_collator(tok, model):
    pad_id = tok.pad_token_id
    def collate(features):
        ids = [f["input_ids"] for f in features]
        lbl = [f["labels"] for f in features]
        L = max(len(x) for x in ids)
        pad = lambda x, v: x + [v]*(L-len(x))
        batch_ids = [pad(x, pad_id) for x in ids]
        batch_mask = [[1]*len(x)+[0]*(L-len(x)) for x in ids]
        batch_lbl = [pad(x, -100) for x in lbl] # 不足分は -100 で穴埋めする（CrossEntropyLossは-100を無視）
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_lbl, dtype=torch.long),
        }
    return collate


# 訓練用関数
def train(model, tok, ds):
    # 訓練用ライブラリを利用してお手軽学習ができる
    args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        save_strategy="epoch",
        logging_steps=10,   # 10 step ごとにログを作成
        remove_unused_columns=False,
        report_to=None,
        dataloader_pin_memory=False,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=False, bf16=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=data_collator(tok, model),
        processing_class=tok,  # tokenizer 引数は非推奨
    )
    trainer.train()
    return trainer

# model 保存。重みだけでなくトレーナーやトークナイザの保存も必要。
def save(trainer, tok):
    os.makedirs(FINAL_DIR, exist_ok=True)
    trainer.save_model(FINAL_DIR)
    tok.save_pretrained(FINAL_DIR)
    print(f"保存先: {FINAL_DIR}")


# 簡易テスト
def quick_test():
    gen = pipeline("text-generation", model=FINAL_DIR, tokenizer=FINAL_DIR,
                   device_map="auto" if torch.cuda.is_available() else None)
    tok = AutoTokenizer.from_pretrained(FINAL_DIR)
    for i, user_text in enumerate(test_text_list, 1):
        prompt = tok.apply_chat_template(
            [{"role":"user","content":user_text}],
            tokenize=False, add_generation_prompt=True)
        output = gen(prompt, max_new_tokens=64, do_sample=False, return_full_text=False)
        # print(output)     # 一度は生データを確認しておくのが吉
        print(f"{i}. 入力: {user_text}\n   応答: {output[0]['generated_text'].strip()}")


# main
def main():
    print("Gemma3 関西弁ファインチューニング開始")
    set_seed(SEED)
    model, tok = load_model_tok()
    ds = build_dataset(tok)
    print(f"学習サンプル数: {len(ds)}")
    trainer = train(model, tok, ds)
    save(trainer, tok)
    quick_test()
    print("完了")

if __name__ == "__main__":
    main()
