"""What you can ask a local model to do with a transcript.

Presets live here as data rather than as branches in the client, so the prompt
text is testable without a model and the web page can build its own menu from
``GET /tasks`` instead of hardcoding one.

Every task inherits :data:`SAFETY_PREAMBLE`. Those rules are what make a local
model's output trustworthy enough to act on -- they are worth more than any
preset, and ``ask`` needs them most, because a free-text instruction is exactly
where a model starts inventing to satisfy the request.
"""

from dataclasses import dataclass

SAFETY_PREAMBLE = """Ti si pažljivi urednik transkripta govora.

Strogo poštuj ova pravila:
- Ne dodaj činjenice, imena ili riječi koje nisu vjerovatno izgovorene.
- Ispravi samo očigledne greške prepoznavanja, pravopis i interpunkciju.
- Ako nisi gotovo potpuno siguran da je riječ pogrešno prepoznata, NE mijenjaj je.
- Ako neki dio nije jasan, zadrži originalnu riječ i dodaj [nejasno] umjesto nagađanja.
- Zadrži smisao, redoslijed i stil govornika.
- Sve što napišeš mora se dati potkrijepiti transkriptom."""


# The conversation's system message. The rules are the same ones every task
# inherits -- the job is different, so the framing is, but "answer only from the
# transcript" is the same promise as "do not add facts that were not spoken".
#
# It lives on the server rather than being sent by the page, so a browser cannot
# edit the rules out of it.
CHAT_SYSTEM = """Odgovaraš na pitanja o transkriptu koji slijedi.

Strogo poštuj ova pravila:
- Odgovaraj isključivo na osnovu transkripta.
- Ako odgovor nije u transkriptu, reci da toga nema. Nemoj pretpostavljati.
- Ne dodaj činjenice, imena, brojeve ni datume koji nisu izgovoreni.
- Ako je nešto nejasno izgovoreno, reci da je nejasno umjesto da nagađaš.
- Odgovaraj kratko i konkretno.

TRANSKRIPT:
{transcript}"""


@dataclass(frozen=True)
class Task:
    """One thing the model can be asked to do.

    ``fields`` is empty for a free-text task and otherwise pairs the JSON key
    the model is asked for with the label the page shows above it, in the order
    they should be rendered.
    """

    id: str
    label: str
    instruction: str
    fields: tuple[tuple[str, str], ...] = ()

    @property
    def structured(self) -> bool:
        return bool(self.fields)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.fields)


TASKS: tuple[Task, ...] = (
    Task(
        id="refine",
        label="Pročisti i sažmi",
        # Measured against qwen3:8b: this wording keeps cleaned_text faithful,
        # mixed languages and all. Spelling out a per-field language split --
        # original for cleaned_text, chosen language for the rest -- did not
        # work and made unclear_parts worse, listing the whole transcript as
        # unclear. An 8B model does not hold that many constraints at once, so
        # the summary fields may come back in English on a Bosnian recording.
        # Fixing that belongs with a stronger backend, not a longer prompt.
        instruction=(
            "Uredi tekst ispod prema pravilima, pa napravi kratak TL;DR i listu "
            "ključnih tačaka isključivo na osnovu onoga što je rečeno. "
            "U cleaned_text zadrži jezik originala doslovno -- ako govornik "
            "prelazi s jednog jezika na drugi, tako i ostaje. Ne prevodi ga."
        ),
        fields=(
            ("cleaned_text", "Uređeni transkript"),
            ("summary", "TL;DR"),
            ("key_points", "Ključne tačke"),
            ("unclear_parts", "Nejasni dijelovi"),
        ),
    ),
    Task(
        id="minutes",
        label="Zapisnik sa sastanka",
        instruction=(
            "Napravi zapisnik sa sastanka. Odvoji ono što je dogovoreno od onoga "
            "što je ostalo otvoreno. Ako neko ime nije jasno izgovoreno, napiši "
            "[nejasno] umjesto da pogađaš ko je šta rekao."
        ),
        fields=(
            ("summary", "O čemu se radilo"),
            ("decisions", "Dogovoreno"),
            ("action_items", "Zadaci"),
            ("open_questions", "Ostalo otvoreno"),
        ),
    ),
    Task(
        id="actions",
        label="Šta treba uraditi",
        instruction=(
            "Izvuci samo konkretne zadatke iz transkripta. Svaki zadatak napiši "
            "kao jednu rečenicu. Ako je rečeno ko ga radi ili do kada, dodaj to; "
            "ako nije, nemoj izmišljati."
        ),
        fields=(
            ("action_items", "Zadaci"),
            ("open_questions", "Nejasno ili nedovršeno"),
        ),
    ),
    Task(
        id="unify",
        label="Složi misli u jednu bilješku",
        instruction=(
            "Govornik prelazi između jezika i vraća se na iste teme više puta. "
            "Složi to u jednu povezanu bilješku: grupiši ono što ide zajedno, "
            "zadrži redoslijed misli gdje ima smisla, i prevedi sve na traženi "
            "jezik. Ne skraćuj toliko da se izgubi šta je tačno rečeno."
        ),
    ),
    Task(
        id="email",
        label="Napiši kao email",
        instruction=(
            "Napiši email na osnovu transkripta. Samo ono što je rečeno, bez "
            "izmišljenih detalja i bez obećanja koja se ne pominju."
        ),
    ),
    Task(
        id="ask",
        label="Pitaj bilo šta",
        instruction="",
    ),
)

BY_ID = {task.id: task for task in TASKS}
DEFAULT_TASK_ID = "refine"

# What the answer can be written in, independent of what was spoken. Naming the
# language in the prompt is the whole point of "unify": the recording switches
# between three of them and the note has to land in one.
OUTPUT_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("", "Isti jezik kao snimak"),
    ("bs", "bosanski"),
    ("en", "engleski"),
    ("no", "norveški"),
)

OUTPUT_LANGUAGE_NAMES = dict(OUTPUT_LANGUAGES)


def get(task_id: str) -> Task:
    """Look up a task, or raise ``ValueError`` naming the ones that exist."""
    try:
        return BY_ID[task_id]
    except KeyError:
        available = ", ".join(sorted(BY_ID))
        raise ValueError(f"Nepoznata radnja {task_id!r}. Dostupne su: {available}.") from None
