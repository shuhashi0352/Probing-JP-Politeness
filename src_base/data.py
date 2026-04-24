import requests
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

import json
from pathlib import Path

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def pull_data(cfg):
    """
    Pull and save data from Keico corpus to a .csv file
    """

    url = cfg["data"]["url"]
    file_name = url.split('/')[-1]

    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)

    if not os.path.exists(file_path):
        print("\nDownloading data...\n")
        with requests.get(url) as r:
            with open(file_path, 'wb') as f:
                f.write(r.content)
        
        print('Download successful')
    
    else:
        print("\nData already exists. Skip downloading...\n")
    
    return file_path

def read_data(file_path):
    return pd.read_csv(file_path)

def split_df(cfg, df):

    text = cfg["data"]["text_col"]
    label = cfg["data"]["label_col"]
    train_size = cfg["experiment"]["train_size"]
    ratio_dev_test = cfg["experiment"]["ratio_dev_test"]
    seed = cfg["experiment"]["seed"]

    # Raise Error if data lacks any required columns
    missing = [c for c in [text, label] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    train, temporary = train_test_split(df, train_size=train_size, random_state=seed, stratify=df[label])
    test, dev = train_test_split(temporary, train_size=ratio_dev_test, random_state=seed, stratify=temporary[label])

    return train, dev, test, text, label

def split_donor_receiver_df(df, label_col, donor_label=0, receiver_label=3):

    # donor/receiver extraction
    donor_df = df[df[label_col] == donor_label].copy()
    receiver_df = df[df[label_col] == receiver_label].copy()

    # Sanity check
    if len(donor_df) == 0:
        raise ValueError(f"No donor instances found for label={donor_label}")
    if len(receiver_df) == 0:
        raise ValueError(f"No receiver instances found for label={receiver_label}")

    return donor_df, receiver_df # Don't return the base df since it won't be used for causality tests


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

    print(converted_min)

    return converted_full, converted_min



    


    
