# small but separable fixtures shared across tests: two disjoint vocabularies so the
# baseline pipeline reliably learns the split. not a test module.

HUMAN = [
    "i honestly think today felt kinda weird",
    "my friend really liked that messy stuff",
    "honestly i feel weird about today",
    "really messy kinda weird stuff today",
    "my friend felt honestly weird",
    "i really think today felt messy",
]
AI = [
    "furthermore the system facilitates optimal structured frameworks",
    "the comprehensive robust framework facilitates optimal output",
    "moreover structured frameworks facilitate robust comprehensive systems",
    "optimal structured systems facilitate comprehensive frameworks",
    "the robust system facilitates structured optimal output",
    "comprehensive frameworks facilitate moreover optimal systems",
]

TEXTS = HUMAN + AI
LABELS = [0] * len(HUMAN) + [1] * len(AI)
