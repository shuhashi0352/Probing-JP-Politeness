import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
import torch

from probing.extract_hs import build_context_quote_masks

class PolitenessDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

class DASPolitenessDataset(Dataset):
    def __init__(self, encodings, labels, context_masks, quote_masks):
        self.encodings = encodings
        self.labels = labels
        self.context_masks = torch.tensor(context_masks, dtype=torch.long)
        self.quote_masks = torch.tensor(quote_masks, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["context_mask"] = self.context_masks[idx]
        item["quote_mask"] = self.quote_masks[idx]
        return item

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
# Freeze the encoder parameters
def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    
def prepare_model(cfg, train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels):

    bert = cfg["model"]
    batch_size = cfg["task"]["batch_size"]
    LineDistilBERT = bert["name"]
    num_labels = bert["num_labels"]
    seed = cfg["experiment"]["seed"]

    g = torch.Generator()
    g.manual_seed(seed)

    # Create datasets
    train_dataset = PolitenessDataset(train_enc, train_labels)
    dev_dataset = PolitenessDataset(dev_enc, dev_labels)
    test_dataset = PolitenessDataset(test_enc, test_labels)

    # Create DataLoaders for batch training
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, generator=g)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(LineDistilBERT, num_labels=num_labels, output_hidden_states=True)
    freeze_model(model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_num_layers = len(model.distilbert.transformer.layer)

    print("\nModel successfully set up\n")

    return train_dataloader, dev_dataloader, test_dataloader, model, device, model_num_layers

def make_dataloader(enc, labels, cfg, shuffle=False):
    ds = PolitenessDataset(enc, labels)
    return DataLoader(ds, batch_size=cfg["task"]["batch_size"], shuffle=shuffle)


######### DAS ###########


def format_prompt(data, role_map=None):
    role_map = {
        "actor": "俳優",
        "apprentice": "徒弟",
        "athlete": "選手",
        "audience_member": "観客",
        "band_member": "バンドメンバー",
        "bandleader": "バンドリーダー",
        "bank_customer": "銀行の客",
        "bank_teller": "銀行員",
        "bar_manager": "バーの店長",
        "bartender": "バーテンダー",
        "booth_staff_member": "ブーススタッフ",
        "call_in_customer": "電話予約の客",
        "camera_assistant": "撮影助手",
        "camp_counselor": "キャンプ指導員",
        "camper": "キャンプ参加者",
        "chef": "シェフ",
        "choir_director": "合唱団の指揮者",
        "choir_member": "合唱団員",
        "client": "依頼人",
        "club_member": "部員",
        "club_officer": "部の役員",
        "coach": "コーチ",
        "commanding_officer": "上官",
        "committee_member": "委員",
        "concertgoer": "コンサート来場者",
        "courier": "配達員",
        "customer": "客",
        "customer_support_agent": "カスタマーサポート担当者",
        "dance_instructor": "ダンス講師",
        "debate_coach": "ディベート指導者",
        "debate_team_member": "ディベート部員",
        "delivery_driver": "配送ドライバー",
        "delivery_recipient": "荷物の受取人",
        "diner": "食事客",
        "director": "監督",
        "dorm_committee_chair": "寮委員長",
        "driver": "運転手",
        "early_employee": "初期メンバー社員",
        "employee": "店員",
        "event_attendee": "イベント参加者",
        "event_coordinator": "イベントコーディネーター",
        "event_staff_member": "イベントスタッフ",
        "father": "父",
        "father_in_law": "義父",
        "festival_attendee": "祭りの参加者",
        "film_director": "映画監督",
        "fitness_instructor": "フィットネス指導員",
        "flight_attendant": "客室乗務員",
        "I": "私",
        "Hanako_my_classmate": "同級生の花子",
        "friends_parent": "友だちの親",
        "front_desk_clerk": "フロント係",
        "grandchild": "孫",
        "grandmother": "祖母",
        "gym_member": "ジム会員",
        "homeroom_teacher": "担任教師",
        "host": "案内係",
        "hotel_guest": "宿泊客",
        "hr_staff_member": "人事担当者",
        "intern": "インターン",
        "junior_employee": "若手社員",
        "junior_intern": "後輩インターン",
        "kindergarten_student": "幼稚園児",
        "kitchen_staff_member": "厨房スタッフ",
        "lawyer": "弁護士",
        "librarian": "司書",
        "library_patron": "図書館利用者",
        "manager": "マネージャー",
        "master_craftsperson": "親方",
        "museum_visitor": "美術館来館者",
        "new_employee": "新入社員",
        "new_lab_member": "新しく入った研究室メンバー",
        "newspaper_editor": "新聞編集者",
        "nurse": "看護師",
        "older_family_friend": "年上の家族ぐるみの知人",
        "older_roommate": "年上のルームメイト",
        "older_sister": "姉",
        "parent": "保護者",
        "parishioner": "檀家",
        "parking_attendant": "駐車場係員",
        "part_time_staff_member": "アルバイトスタッフ",
        "passenger": "乗客",
        "patient_family_member": "患者の家族",
        "peer_age_teammate": "同年代のチームメイト",
        "pet_owner": "飼い主",
        "professor": "教授",
        "project_lead": "プロジェクトリーダー",
        "regular_customer": "常連客",
        "research_assistant": "研究補助者",
        "research_supervisor": "研究指導者",
        "reservation_agent": "予約担当者",
        "resident_assistant": "寮の指導員",
        "resident_student": "寮生",
        "restaurant_guest": "来店客",
        "restaurant_manager": "飲食店の店長",
        "sales_associate": "販売員",
        "senior_employee": "先輩社員",
        "senior_intern": "先輩インターン",
        "senior_lab_member": "先輩研究室メンバー",
        "shift_supervisor": "シフト責任者",
        "shopper": "買い物客",
        "soldier": "兵士",
        "son": "息子",
        "son_in_law": "義息子",
        "staff_writer": "記者",
        "startup_founder": "スタートアップ創業者",
        "store_clerk": "店員",
        "store_manager": "店長",
        "student": "学生",
        "studio_member": "スタジオ会員",
        "subscriber": "契約者",
        "team_captain": "キャプテン",
        "team_member": "チームメンバー",
        "teammate": "チームメイト",
        "temple_office_staff_member": "寺務所の職員",
        "tour_guide": "ツアーガイド",
        "tourist": "観光客",
        "trainee": "研修生",
        "trainer": "指導担当者",
        "usher": "案内係",
        "venue_staff_member": "会場スタッフ",
        "veterinary_receptionist": "動物病院の受付",
        "volunteer": "ボランティア",
        "volunteer_leader": "ボランティアリーダー",
        "volunteer_member": "ボランティアメンバー",
        "waiter": "ウェイター",
        "younger_family_friend": "年下の家族ぐるみの知人",
        "younger_roommate": "年下のルームメイト",
        "younger_sister": "妹",
        "younger_teammate": "年下のチームメイト",
    }
    
    speaker_role = data["context"]["speaker_role"]

    listener_role = data["context"]["listener_role"]

    speaker = role_map.get(speaker_role, speaker_role)

    listener = role_map.get(listener_role, listener_role)

    utterance = data["utterance"].strip("“”\"「」")

    return f"{speaker}は{listener}に「{utterance}」と言った。"

def convert_example(data, role_map=None):

    return {

        "id": data["id"],

        "text": format_prompt(data, role_map=role_map),

        "label": data["naturalness"],

        "realized_level": data["realized_level"],

        "expected_level": data["expected_level"],

        "speaker_role": data["context"]["speaker_role"],

        "listener_role": data["context"]["listener_role"],

        "setting": data["context"]["setting"],

    }

def convert_dataset(cfg, role_map=None):
    input_path = cfg["data"]["binary_data"]
    input_path = Path(input_path)

    output_full_path = cfg["data"]["binary_full_out"]
    output_full_path = Path(output_full_path)

    output_min_path = cfg["data"]["binary_min_out"]
    output_min_path = Path(output_min_path)

    with input_path.open("r", encoding="utf-8") as f:

        dataset = json.load(f)

    converted_full = [convert_example(item, role_map=role_map) for item in dataset]

    converted_min = [

        {
            "id": item["id"],
            "text": item["text"],
            "label": item["label"]
        }

        for item in converted_full

    ]

    with output_full_path.open("w", encoding="utf-8") as f:

        json.dump(converted_full, f, ensure_ascii=False, indent=2)

    with output_min_path.open("w", encoding="utf-8") as f:

        json.dump(converted_min, f, ensure_ascii=False, indent=2)

    return converted_full, converted_min


def load_politeness_json(json_path):
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of examples.")

    df = pd.DataFrame(data)

    required_cols = ["id", "text", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def add_label_ids(df):
    """
    Add integer labels for modeling.
    """
    label2id = {
        "unnatural": 0,
        "natural": 1,
    }

    unknown = sorted(set(df["label"]) - set(label2id))
    if unknown:
        raise ValueError(f"Unknown labels found: {unknown}")

    df = df.copy()
    df["label_id"] = df["label"].map(label2id)

    return df, label2id


def add_utterance_group_id(df):
    df = df.copy()

    def extract_utterance(text):
        start = text.find("「")
        end = text.find("」")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Invalid text format: {text}")
        return text[start + 1:end]

    df["utterance"] = df["text"].apply(extract_utterance)
    df["utterance_group_id"] = pd.factorize(df["utterance"])[0] + 1

    return df


def split_by_utterance_group(
    df,
    train_size=0.70,
    dev_size=0.15,
    test_size=0.15,
    seed=42,
):
    total = train_size + dev_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train/dev/test must sum to 1.0, got {total}")

    if "utterance_group_id" not in df.columns:
        raise ValueError("Missing utterance_group_id. Run add_utterance_group_id(df) first.")

    groups = sorted(df["utterance_group_id"].unique())

    train_groups, temp_groups = train_test_split(
        groups,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
    )

    relative_dev_size = dev_size / (dev_size + test_size)

    dev_groups, test_groups = train_test_split(
        temp_groups,
        train_size=relative_dev_size,
        random_state=seed,
        shuffle=True,
    )

    train_groups = set(train_groups)
    dev_groups = set(dev_groups)
    test_groups = set(test_groups)

    train_df = df[df["utterance_group_id"].isin(train_groups)].copy()
    dev_df = df[df["utterance_group_id"].isin(dev_groups)].copy()
    test_df = df[df["utterance_group_id"].isin(test_groups)].copy()

    _check_no_utterance_group_leakage(train_df, dev_df, test_df)
    _report_split_stats(train_df, dev_df, test_df)

    return train_df, dev_df, test_df


def _check_no_utterance_group_leakage(train_df, dev_df, test_df):
    train_groups = set(train_df["utterance_group_id"])
    dev_groups = set(dev_df["utterance_group_id"])
    test_groups = set(test_df["utterance_group_id"])

    if train_groups & dev_groups:
        raise ValueError("Utterance leakage found between train and dev.")
    if train_groups & test_groups:
        raise ValueError("Utterance leakage found between train and test.")
    if dev_groups & test_groups:
        raise ValueError("Utterance leakage found between dev and test.")


def _report_split_stats(train_df, dev_df, test_df):
    for name, split in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        print(f"\n{name.upper()}")
        print(f"rows: {len(split)}")
        print(f"utterance groups: {split['utterance_group_id'].nunique()}")
        print("label counts:")
        print(split["label"].value_counts().sort_index())


def split_data(cfg):
    json_path = cfg["data"]["binary_min_out"]
    text_col = cfg["data"].get("text_col", "text")
    label_col = cfg["data"].get("label_col", "label")

    train_size = cfg["experiment"].get("train_size", 0.70)
    dev_size = cfg["experiment"].get("dev_size", 0.15)
    test_size = cfg["experiment"].get("test_size", 0.15)
    seed = cfg["experiment"].get("seed", 42)

    df = load_politeness_json(json_path)

    if text_col not in df.columns:
        raise ValueError(f"Text column not found: {text_col}")

    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    # Standardize internal column names if needed
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})

    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    df, label2id = add_label_ids(df)
    df = add_utterance_group_id(df)

    train_df, dev_df, test_df = split_by_utterance_group(
        df,
        train_size=train_size,
        dev_size=dev_size,
        test_size=test_size,
        seed=seed,
    )

    return train_df, dev_df, test_df, label2id

def make_aligned_das_dfs(split_df):
    receiver_rows = []
    donor_rows = []

    for _, group in split_df.groupby("utterance_group_id"):
        natural = group[group["label_id"] == 1]
        unnatural = group[group["label_id"] == 0]

        if len(natural) == 0 or len(unnatural) == 0:
            continue

        n = min(len(natural), len(unnatural))

        natural = natural.sample(n=n, random_state=42).reset_index(drop=True)
        unnatural = unnatural.sample(n=n, random_state=42).reset_index(drop=True)

        # unnatural receiver -> natural donor
        receiver_rows.extend(unnatural.to_dict("records"))
        donor_rows.extend(natural.to_dict("records"))

        # natural receiver -> unnatural donor
        receiver_rows.extend(natural.to_dict("records"))
        donor_rows.extend(unnatural.to_dict("records"))

    receiver_df = pd.DataFrame(receiver_rows).reset_index(drop=True)
    donor_df = pd.DataFrame(donor_rows).reset_index(drop=True)

    return receiver_df, donor_df

def encode_df(df, tokenizer, max_length=128):
    return tokenizer(
        df["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

def make_labels(df):
    return torch.tensor(df["label_id"].tolist(), dtype=torch.long)

def make_das_dataloaders(split_df, tokenizer, batch_size=16, max_length=128):
    receiver_df, donor_df = make_aligned_das_dfs(split_df)

    receiver_enc = encode_df(receiver_df, tokenizer, max_length=max_length)
    donor_enc = encode_df(donor_df, tokenizer, max_length=max_length)

    receiver_labels = make_labels(receiver_df)
    donor_labels = make_labels(donor_df)

    receiver_context_masks, receiver_quote_masks = build_context_quote_masks(
        receiver_df["text"].tolist(),
        receiver_enc,
        tokenizer,
    )

    donor_context_masks, donor_quote_masks = build_context_quote_masks(
        donor_df["text"].tolist(),
        donor_enc,
        tokenizer,
    )

    receiver_ds = DASPolitenessDataset(
        receiver_enc,
        receiver_labels,
        receiver_context_masks,
        receiver_quote_masks,
    )

    donor_ds = DASPolitenessDataset(
        donor_enc,
        donor_labels,
        donor_context_masks,
        donor_quote_masks,
    )

    receiver_dl = DataLoader(receiver_ds, batch_size=batch_size, shuffle=False)
    donor_dl = DataLoader(donor_ds, batch_size=batch_size, shuffle=False)

    return receiver_dl, donor_dl, receiver_df, donor_df
    
