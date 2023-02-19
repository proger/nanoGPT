import json
import csv
from tqdm import tqdm
import argparse
from rouge import Rouge
from typing import List


def get_jaccard_sim(hypothesis: List[str], reference: List[str]) -> float:
    a = set(map(str.lower, reference))
    b = set(map(str.lower, hypothesis))
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_tags_accuracy(hypothesis: List[str], reference: List[str]) -> float:
    a = set(map(str.lower, reference))
    b = set(map(str.lower, hypothesis))
    c = a.intersection(b)

    return float(len(c) / len(a))


docs = [
    {
        "_id": "4c3a9b88a54b0bac0a1f57c2eee1e1f4fb14d26c",
        "date_of_publish": {"$date": "2023-02-02T09:19:00Z"},
        "source_info": {"slug": "eurointegration_com_ua"},
        "tags": ["Швейцарія", "Війна з РФ"],
        "text": 'Провідні політики у Швейцарії закликають продати майже 100 законсервованих\nтанків Leopard 2 за символічну плату Польщі, Словаччині та Чехії замість\nтанків, які їхні уряди планують відправити в Україну.\n\nПро це пише Bloomberg, повідомляє "Європейська правда".\n\nУ бункері в гірській місцевості на сході Швейцарії майже десять років стоять\nзаконсервовані майже 100 танків Leopard 2, призначені для списання, оскільки\nвважалося, що бронетанкова війна в Європі залишилася в минулому.\n\nРішення Німеччини та інших європейських держав відправлять свої танки в\nУкраїну активізувало дебати в Берні про те, як і коли нейтральна країна може\nпосилити власну допомогу.\n\nПро постачання зброї безпосередньо в Україну не йдеться, але є інші варіанти.\n"Війна змінила дискусію в Швейцарії. Ми не можемо відмовитися від\nшвейцарського нейтралітету, але ми повинні говорити про те, які можливості ми\nмаємо для підтримки країн, що відстоюють ті ж демократичні цінності, що і\nШвейцарія", - сказала Майя Рінікер, член парламенту від центристської і\nпробізнесової партії СвДП, яка внесла пропозицію про передачу танків.\n\nРінікер закликала офіційно вивести танки Leopard з експлуатації, а потім\nпродати їх за символічну ціну в один франк східним сусідам Швейцарії за умови,\nщо ці танки ніколи не будуть реекспортовані в Україну. За її словами,\nШвейцарії не потрібні всі танки, і хоча її пропозиція була відхилена в першому\nчитанні минулого тижня, вона прагне винести її на повторне голосування цієї\nвесни.\n\n96 танків, що зберігаються у Швейцарії, становитимуть майже третину з 300\nтанків, які, за словами президента України Володимира Зеленського, необхідні\nйому для того, щоб переломити хід війни з Росією. Машини були спочатку\nзаплановані до продажу в 2014 році на підставі того, що вони були\nнадлишковими.\n\nБудь-який продаж танків вимагатиме їх масштабного виведення з експлуатації, а\nотже, і відповідної резолюції парламенту, повідомила речниця Міністерства\nоборони Швейцарії. Наразі Швейцарія не отримала жодного запиту від Німеччини\nпро продаж частини танків виробнику Rheinmetall AG, а також не має жодних\nзапитів від інших країн, які перебувають на розгляді, повідомила вона.\n\nУ четвер швейцарські депутати також обговорюватимуть законопроєкт, який\nнабирає певної популярності в парламенті, і який дозволить Німеччині, Іспанії\nта Данії також відправляти на фронт боєприпаси швейцарського виробництва.\n\nНагадаємо, парламентська комісія **затвердила подання, яке дозволить третім\nкраїнам реекспортувати зброю** в Україну, оскільки вторгнення Росії порушило\nміжнародне право. Однак це рішення ще остаточно не затверджене, а перебуває на\nпочатковій стадії.\n\nШвейцарія **зіткнулася з критикою Німеччини та Іспанії** через блокування\nпоставок швейцарських боєприпасів в Україну.\n\nЧитайте також: **Крига скресла: як нейтральна Швейцарія схиляється до\nреекспорту озброєнь в Україну**.',
        "text_length": 2947,
        "title": 'У Швейцарії пропонують продати 96 "зайвих" Leopard країнам, які поставляють\nтанки Україні',
        "url": "https://www.eurointegration.com.ua/news/2023/02/2/7155366/",
        "word_count": 489,
        "generated_tags": ["Війна з РФ", "Швейцарія", "Німеччина"],
        "generated_title": "У Швейцарії не можуть пробачити Німеччині, що вона продає танки Україні",
    }
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate_generated")
    parser.add_argument("infile", type=argparse.FileType("r"), help="JSONL files with original and generated tags and titles")
    parser.add_argument("outfile", type=argparse.FileType("w"), help="CSV file with the results")
    args = parser.parse_args()

    rouge = Rouge()

    w = csv.DictWriter(args.outfile, fieldnames=["_id", "tags_jaccard", "tags_accuracy", "rouge-1_f", "rouge-1_p", "rouge-1_r", "rouge-2_f", "rouge-2_p", "rouge-2_r", "rouge-l_f", "rouge-l_p", "rouge-l_r"])
    
    w.writeheader()    
    for doc in tqdm(map(json.loads, args.infile)):
        rouge_scores = rouge.get_scores(doc["generated_title"], doc["title"])

        w.writerow({
            "_id": doc["_id"],
            "tags_jaccard": get_jaccard_sim(hypothesis=doc["generated_tags"], reference=doc["tags"]),
            "tags_accuracy": get_tags_accuracy(hypothesis=doc["generated_tags"], reference=doc["tags"]),
            "rouge-1_f": rouge_scores[0]["rouge-1"]["f"],
            "rouge-1_p": rouge_scores[0]["rouge-1"]["p"],
            "rouge-1_r": rouge_scores[0]["rouge-1"]["r"],
            "rouge-2_f": rouge_scores[0]["rouge-2"]["f"],
            "rouge-2_p": rouge_scores[0]["rouge-2"]["p"],
            "rouge-2_r": rouge_scores[0]["rouge-2"]["r"],
            "rouge-l_f": rouge_scores[0]["rouge-l"]["f"],
            "rouge-l_p": rouge_scores[0]["rouge-l"]["p"],
            "rouge-l_r": rouge_scores[0]["rouge-l"]["r"],
        })
