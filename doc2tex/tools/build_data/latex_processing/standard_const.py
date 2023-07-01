import re

MIN_CHARS = 30
MAX_CHARS = 1000
MIN_TOKENS = 5

DOLLAR = re.compile(
    r"((?<!\$)\${2}(?!\$))(.{%i,%i}?)(?<!\\)(?<!\$)\1(?!\$)" % (1, MAX_CHARS)
)

EQUATION = re.compile(
    r"\\begin\{(equation|displaymath)\*?\}(.{%i,%i}?)\\end\{\1\*?\}" % (1, MAX_CHARS),
    re.S,
)

ALIGN = re.compile(
    r"(\\begin\{(align|alignedat|alignat|flalign|eqnarray|gather|gathered|alignedat)\*?\}(.{%i,%i}?)\\end\{\2\*?\})"
    % (1, MAX_CHARS),
    re.S,
)

DISPLAYMATH = re.compile(
    r"(?:\\displaystyle)(.{%i,%i}?)((?<!\\)\}?(?:\"|<))" % (1, MAX_CHARS), re.S
)

OUTER_WHITESPACE = re.compile(
    r"^\\,|\\,$|"
    r"^~|~$|"
    r"^\\ |\\ $|"
    r"^\\thinspace|\\thinspace$|"
    r"^\\medspace|\\medspace$|"
    r"^\\thickspace|\\thickspace$|"
    r"^\\!|\\!$|"
    r"^\\:|\\:$|"
    r"^\\;|\\;$|"
    r"^\\enspace|\\enspace$|"
    r"^\\quad|\\quad$|"
    r"^\\qquad|\\qquad$|"
    r"^\\hspace\*?{[a-zA-Z0-9]+}|\\hspace\*?{[a-zA-Z0-9]+}$|"
    r"^\\hfill|\\hfill$|"
    r"^\\kern{[a-zA-Z0-9]+}|\\kern{[a-zA-Z0-9]+}$|"
    r"^\\mkern{[a-zA-Z0-9]+}|\\mkern{[a-zA-Z0-9]+}$|"
    r"^\\mskip{[a-zA-Z0-9]+}|\\mskip{[a-zA-Z0-9]+}$|"
    r"^\\phantom{[a-zA-Z0-9]+}|\\phantom{[a-zA-Z0-9]+}$|"
    r"^\\hphantom{[a-zA-Z0-9]+}|\\hphantom{[a-zA-Z0-9]+}$|"
    r"^\\vphantom{[a-zA-Z0-9]+}|\\vphantom{[a-zA-Z0-9]+}$|"
    r"^\\negthinspace|\\negthinspace$|"
    r"^\\negmedspace|\\negmedspace$|"
    r"^\\negthickspace|\\negthickspace$|"
    r"^\\mathstrut|\\mathstrut$|"
)

WHITESPACE = re.compile(
    r"\\,|~|\\thinspace|\\medspace|\\thickspace|\\!|\\:|\\;|\\enspace|\\quad|\\qquad|\\hspace\*?{[a-zA-Z0-9]+}|"
    r"\\hfill|\\kern{[a-zA-Z0-9]+}|\\mkern{[a-zA-Z0-9]+}|\\mskip{[a-zA-Z0-9]+}|"
    r"\\phantom{[a-zA-Z0-9]+}|\\hphantom{[a-zA-Z0-9]+}|\\vphantom{[a-zA-Z0-9]+}|"
    r"\\negthinspace|\\negmedspace|\\negthickspace|\\mathstrut"
)

NO_TAG = [r"\\notag", r"\\nonumber", r"\\noalign{.*}"]

LABEL_TAGS = [
    re.compile(r"\\%s\s?\{(.*?)\}" % s) for s in ["ref", "cite", "label", "eqref"]
]

OPERATORS = [
    "arccos",
    "arcsin",
    "arctan",
    "arctg",
    "arcctg",
    "arg",
    "cos",
    "cosh",
    "cot",
    "coth",
    "csc",
    "deg",
    "det",
    "dim",
    "exp",
    "gcd",
    "hom",
    "inf",
    "injlim",
    "ker",
    "lg",
    "lim",
    "liminf",
    "limsup",
    "ln",
    "log",
    "max",
    "min",
    "Pr",
    "projlim",
    "sec",
    "sin",
    "sinh",
    "sup",
    "tan",
    "tanh",
    "ch",
    "argmax",
    "argmin",
    "cosec",
    "cotg",
    "ctg",
    "cth",
    "plim",
    "sh",
    "tg",
    "th",
]


# NEEDBRACKET_OP = ['\sum', '\frac', '\cfrac', '\sfrac', '\limits', '\nolimits', '\iint', '\iiint', '\oint', '\int', '\prod', '\coprod', '\intop', '\smallint', '\oiint', '\bigotimes', '\binom', '\idotsintm', '\biguplus', '\bigwedge', '\bigcap', '\bigsqcup', '\bigcup', '\iiiint']


SKIP_TOK = [r"\\smallskip", r"\\medskip", r"\\bigskip", r"\\nomallineskiplimit"]

STANDARD_SPACE = {
    2: 1,
    4: 2,
    8: 4,
}

STANDARD_WHITESPACE_SPACE = "\\,"

FONT = {
    r"\textit": r"\mathit",
    r"\textbf": r"\mathbf",
    r"\textrm": r"\mathrm",
    r"\textsf": r"\mathsf",
    r"\textnormal": r"\mathnormal",
}

SIZE = [
    r"\Huge",
    r"\huge",
    r"\LARGE",
    r"\Large",
    r"\large",
    r"\small",
    r"\normalsize",
    r"\footnotesize",
    r"\scriptsize",
    r"\tiny",
]
