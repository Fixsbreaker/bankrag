"""Сверка владельцев счетов с перечнями АФМ РК.

Перечни скачивает scripts/download_lists.py в data/raw/lists/ (открытый API api.websfm.kz).
Сверка идёт локально, без интернета:

1. ИИН/БИН совпал — совпадение по ИИН/БИН (в записях API ИИН сейчас обычно нет, но поле поддерживается).
2. Организация: совпало название после нормализации.
3. Человек: совпали ФИО и дата рождения. Если в перечне даты нет, совпадение помечается как
   «по ФИО, дату рождения проверить». Если даты есть и различаются, это однофамилец, совпадения нет.

4. Похожее написание: опечатка, латиница вместо кириллицы, «Қайрат» / «Кайрат» / «Kairat» / «Qairat».
   ФИО приводится к латинскому ключу, затем сравнивается RapidFuzz (token_sort_ratio). Такое совпадение
   всегда только «проверить вручную» (оранжевый), а не красный: похожее имя даёт ложные тревоги.
   Если в перечне и у владельца есть даты рождения и они различаются, это однофамилец, совпадения нет.

Кроме настоящих перечней загружается configs/watchlist_demo.json: одна вымышленная тестовая запись
для демо. Реальным лицам из перечней синтетические переводы не приписываются.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz, process

LISTS_DIR = "data/raw/lists"
DEMO_FILE = "configs/watchlist_demo.json"
DEMO_CATEGORY = "demo"
LIST_TITLES = {
    "sanction_category_1": "Перечень организаций и лиц, связанных с финансированием терроризма и экстремизма",
    "sanction_category_2": "Список лиц, причастных к террористической деятельности",
    "sanction_category_3": "Перечень организаций и лиц, связанных с финансированием распространения оружия "
                           "массового уничтожения (ФРОМУ)",
    "sanction_category_4": "Санкционный перечень Совета Безопасности ООН",
    DEMO_CATEGORY: "Тестовая запись BankRAG (вымышленная, для демо)",
}
SHORT_TITLES = {
    "sanction_category_1": "перечень АФМ: терроризм и экстремизм",
    "sanction_category_2": "перечень АФМ: причастные к террористической деятельности",
    "sanction_category_3": "перечень АФМ: ФРОМУ",
    "sanction_category_4": "перечень СБ ООН",
    DEMO_CATEGORY: "тестовая запись демо",
}
MATCH_IIN = "ИИН/БИН"
MATCH_NAME_DOB = "ФИО и дата рождения"
MATCH_NAME_ONLY = "ФИО, в перечне нет даты рождения — проверить"
MATCH_ORG = "название организации"
MATCH_NAME_QUERY = "ФИО совпало, дату рождения сверить"
MATCH_FUZZY = "похожее написание ({score}%) — проверить вручную"
# Порог сходства всего имени; в приложении берётся из config.yaml (watchlist.fuzzy_threshold).
FUZZY_THRESHOLD = 88
# Каждое слово записи перечня должно найти похожее слово у владельца: иначе «Шакиров Руслан Маратович»
# и «Садыков Руслан Маратович» дают 90% по одинаковым имени и отчеству при разных фамилиях.
WORD_THRESHOLD = 80
PERSON, ORGANIZATION = "person", "organization"

# Казахские буквы сводятся к русским: в разных системах одно имя пишут по-разному («Әнуар» и «Ануар»).
_LETTERS = str.maketrans({"Ә": "А", "Ғ": "Г", "Қ": "К", "Ң": "Н", "Ө": "О", "Ұ": "У", "Ү": "У", "Һ": "Х",
                          "І": "И", "Ё": "Е", "Й": "И"})
# Организационно-правовая форма не входит в сравнение: «ТОО «Альфа»» и «Альфа» — одно название.
_LEGAL_FORMS = {"ТОО", "АО", "ИП", "ООО", "ОАО", "ЗАО", "ПК", "LLP", "LLC", "JSC", "LTD", "INC"}


def normalize_name(text: Any) -> str:
    """Ключ сравнения: верхний регистр, без кавычек, знаков, формы собственности; слова по алфавиту.

    Слова сортируются, потому что в перечнях встречается и «Фамилия Имя», и «Имя Фамилия».
    """
    if text is None:
        return ""
    value = str(text).upper().translate(_LETTERS)
    value = re.sub(r"[^\w]+", " ", value)
    words = [w for w in value.split() if w not in _LEGAL_FORMS]
    return " ".join(sorted(words))


# Кириллица → латиница. Затем варианты латиницы сводятся к одному: Q и K, J и ZH, Y и I.
_TO_LATIN = {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ж": "ZH", "З": "Z", "И": "I",
             "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
             "Ф": "F", "Х": "H", "Ц": "TS", "Ч": "CH", "Ш": "SH", "Щ": "SH", "Ъ": "", "Ы": "Y", "Ь": "", "Э": "E",
             "Ю": "YU", "Я": "YA"}
_LATIN_VARIANTS = [("KH", "H"), ("DZH", "ZH"), ("J", "ZH"), ("Q", "K"), ("W", "V"), ("X", "KS"), ("YE", "E"), ("Y", "I"),
                   ("C", "K")]


def latin_key(text: Any) -> str:
    """Ключ для нечёткого сравнения: нормализованное имя латиницей, варианты букв сведены, без удвоений.

    «Қайрат», «Кайрат», «Kairat» и «Qairat» дают один ключ KAIRAT.
    """
    words = []
    for word in normalize_name(text).split():
        value = "".join(_TO_LATIN.get(ch, ch) for ch in word)
        for old, new in _LATIN_VARIANTS:
            value = value.replace(old, new)
        value = re.sub(r"(.)\1+", r"\1", value)
        if value:
            words.append(value)
    return " ".join(sorted(words))


def _words_match(key: str, other: str) -> bool:
    """Каждое слово более короткого ключа похоже хотя бы на одно слово другого."""
    a, b = key.split(), other.split()
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return all(max(fuzz.ratio(word, candidate) for candidate in long) >= WORD_THRESHOLD for word in short)


def _date(value: Any) -> str | None:
    text = str(value or "").strip()
    return text[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", text) else None


def _digits(value: Any) -> str | None:
    text = re.sub(r"\D", "", str(value or ""))
    return text if len(text) == 12 else None


@dataclass(frozen=True)
class ListRecord:
    """Одна запись перечня в общем виде для обоих форматов API."""

    category: str
    record_id: str
    full_name: str
    kind: str
    date_of_birth: str | None
    added: str | None
    note: str
    iin: str | None = None

    @property
    def list_title(self) -> str:
        return LIST_TITLES.get(self.category, self.category)


def parse_record(category: str, raw: dict) -> ListRecord:
    """Запись из API: у перечней терроризма поля full_name/type, у ФРОМУ и СБ ООН — name/dob/listed_at."""
    name = raw.get("full_name") or raw.get("name") or ""
    # В API встречаются типы «organizationcis» и с переводом строки: всё, что начинается с organization, — организация.
    kind = str(raw.get("type") or "").strip().lower()
    if not kind:
        looks_like_org = re.search(r"[«\"]|КОРПОРАЦ|КОМПАН|ОРГАНИЗАЦ|БАНК|CORPORATION|COMPANY", str(name).upper())
        kind = ORGANIZATION if looks_like_org and not (raw.get("dob") or raw.get("date_of_birth")) else PERSON
    kind = ORGANIZATION if kind.startswith(ORGANIZATION) else PERSON
    return ListRecord(category=category, record_id=str(raw.get("id", "")), full_name=" ".join(str(name).split()),
                      kind=kind, date_of_birth=_date(raw.get("date_of_birth") or raw.get("dob")),
                      added=_date(raw.get("added_to_list") or raw.get("listed_at")), note=str(raw.get("note") or ""),
                      iin=_digits(raw.get("iin")))


@dataclass
class Hit:
    """Совпадение владельца счёта с записью перечня."""

    record: ListRecord
    match: str
    similarity: int | None = None

    @property
    def strict(self) -> bool:
        """Точное совпадение (красный). Похожее написание — только «проверить вручную»."""
        return self.similarity is None

    def as_dict(self) -> dict:
        r = self.record
        return {"list": r.category, "list_title": r.list_title, "list_short": SHORT_TITLES.get(r.category, r.category),
                "record_id": r.record_id, "listed_name": r.full_name, "listed_kind": r.kind,
                "listed_birth_date": r.date_of_birth, "added_to_list": r.added, "match": self.match,
                "is_demo": r.category == DEMO_CATEGORY, "strict": self.strict, "similarity": self.similarity}


@dataclass
class Watchlist:
    """Все загруженные перечни и индексы для быстрой сверки."""

    records: list[ListRecord] = field(default_factory=list)
    downloaded_at: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    fuzzy_threshold: int = FUZZY_THRESHOLD

    def __post_init__(self) -> None:
        self._by_name: dict[str, list[ListRecord]] = {}
        self._by_iin: dict[str, list[ListRecord]] = {}
        self._latin: dict[str, list[ListRecord]] = {}
        for record in self.records:
            key = normalize_name(record.full_name)
            if key:
                self._by_name.setdefault(key, []).append(record)
                self._latin.setdefault(latin_key(record.full_name), []).append(record)
            if record.iin:
                self._by_iin.setdefault(record.iin, []).append(record)
        self._latin_keys = list(self._latin)

    @property
    def real_count(self) -> int:
        return sum(v for k, v in self.counts.items() if k != DEMO_CATEGORY)

    def screen(self, name: Any, kind: str = PERSON, birth_date: Any = None, iin: Any = None) -> list[Hit]:
        """Совпадения для одного владельца. Пустой список — в перечнях не найден."""
        hits: list[Hit] = []
        seen: set[tuple[str, str]] = set()

        def add(record: ListRecord, match: str) -> None:
            if (record.category, record.record_id) not in seen:
                seen.add((record.category, record.record_id))
                hits.append(Hit(record, match))

        number = _digits(iin)
        for record in self._by_iin.get(number, []) if number else []:
            add(record, MATCH_IIN)
        birth = _date(birth_date)
        for record in self._by_name.get(normalize_name(name), []):
            if kind == ORGANIZATION or record.kind == ORGANIZATION:
                if kind == ORGANIZATION and record.kind == ORGANIZATION:
                    add(record, MATCH_ORG)
            elif record.date_of_birth is None:
                add(record, MATCH_NAME_ONLY)
            elif birth is not None and birth == record.date_of_birth:
                add(record, MATCH_NAME_DOB)
        return hits

    def _similar_keys(self, keys: list[str]) -> list[list[tuple[str, int]]]:
        """Для каждого ключа — похожие ключи перечня со сходством. Сравнение пачкой (cdist) на всех ядрах."""
        if not keys or not self._latin_keys:
            return [[] for _ in keys]
        matrix = process.cdist(keys, self._latin_keys, scorer=fuzz.token_sort_ratio,
                               score_cutoff=self.fuzzy_threshold, dtype=np.uint8, workers=-1)
        result = []
        for key, row in zip(keys, matrix):
            found = []
            for j in np.flatnonzero(row):
                other = self._latin_keys[j]
                if key and _words_match(key, other):
                    found.append((other, int(row[j])))
            result.append(found)
        return result

    def _similar_records(self, name: Any) -> list[tuple[ListRecord, int]]:
        """Записи с похожим написанием имени."""
        return [(record, score) for key, score in self._similar_keys([latin_key(name)])[0] for record in self._latin[key]]

    def _filter_similar(self, name: Any, kind: str, birth_date: Any,
                        similar: list[tuple[str, int]]) -> list[Hit]:
        exact = {(r.category, r.record_id) for r in self._by_name.get(normalize_name(name), [])}
        birth = _date(birth_date)
        hits = []
        for key, score in similar:
            for record in self._latin[key]:
                if (record.category, record.record_id) in exact or record.kind != kind:
                    continue
                if kind == PERSON and birth and record.date_of_birth and birth != record.date_of_birth:
                    continue
                hits.append(Hit(record, MATCH_FUZZY.format(score=score), score))
        return hits

    def screen_similar(self, name: Any, kind: str = PERSON, birth_date: Any = None) -> list[Hit]:
        """Похожие записи для владельца: только «проверить вручную».

        Тип (человек или организация) должен совпасть. Разные известные даты рождения — однофамилец.
        Записи, уже совпавшие точно, сюда не попадают.
        """
        return self._filter_similar(name, kind, birth_date, self._similar_keys([latin_key(name)])[0])

    def screen_similar_many(self, rows: list[tuple[Any, str, Any]]) -> list[list[Hit]]:
        """То же для многих владельцев сразу: (имя, тип, дата рождения) → похожие записи."""
        found = self._similar_keys([latin_key(name) for name, _, _ in rows])
        return [self._filter_similar(name, kind, birth, similar) for (name, kind, birth), similar in zip(rows, found)]

    def lookup(self, query: Any) -> list[Hit]:
        """Поиск по запросу аналитика: ИИН/БИН или ФИО/название без даты рождения."""
        number = _digits(query)
        if number:
            return [Hit(r, MATCH_IIN) for r in self._by_iin.get(number, [])]
        exact = [Hit(r, MATCH_ORG if r.kind == ORGANIZATION else MATCH_NAME_QUERY)
                 for r in self._by_name.get(normalize_name(query), [])]
        seen = {(h.record.category, h.record.record_id) for h in exact}
        similar = [Hit(r, MATCH_FUZZY.format(score=score), score) for r, score in self._similar_records(query)
                   if (r.category, r.record_id) not in seen]
        return exact + similar


def load_watchlist(root: Path, include_demo: bool = True, fuzzy_threshold: int = FUZZY_THRESHOLD) -> Watchlist:
    """Перечни из data/raw/lists/ и тестовая запись демо. Если перечни не скачаны, остаётся только демо."""
    records: list[ListRecord] = []
    counts: dict[str, int] = {}
    downloaded_at = None
    lists_dir = root / LISTS_DIR
    meta_path = lists_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        downloaded_at = meta.get("downloaded_at")
        for category in meta.get("lists", {}):
            path = lists_dir / f"{category}.json"
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                records += [parse_record(category, r) for r in raw]
                counts[category] = len(raw)
    demo_path = root / DEMO_FILE
    if include_demo and demo_path.exists():
        raw = json.loads(demo_path.read_text(encoding="utf-8"))
        records += [parse_record(DEMO_CATEGORY, r) for r in raw]
        counts[DEMO_CATEGORY] = len(raw)
    return Watchlist(records, downloaded_at, counts, fuzzy_threshold)
