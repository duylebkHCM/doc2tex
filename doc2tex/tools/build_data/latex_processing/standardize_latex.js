const path = require('path');
var katex = require(path.join(__dirname,"third_party/katex/katex.js"))
options = require(path.join(__dirname,"third_party/katex/src/Options.js"))
var readline = require('readline');
var rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});


rl.on('line', function(line){
    a = line
    if (line[0] == "%") {
        line = line.substr(1, line.length - 1);
    }
    line = line.split('%')[0];

    line = line.split('\\~').join(' ');

    for (var i = 0; i < 300; i++) {
        line = line.replace(/\\>/, " "); //replace space token can also update at group.spacing function
        line = line.replace('$', ' ');
        line = line.replace(/\\label{.*?}/, "");
    }

    if (line.indexOf("matrix") == -1 && line.indexOf("cases")==-1 &&
        line.indexOf("array")==-1 && line.indexOf("begin")==-1)  {
        for (var i = 0; i < 300; i++) {
            line = line.replace(/\\\\/, "\\,");
        }
    }

    line = line + " "
    // global_str is tokenized version (build in parser.js)
    // norm_str is normalized version build by renderer below.
    try {


        if (process.argv[2] == "tokenize") {
            var tree = katex.__parse(line, {});
            console.log(global_str.replace(/\\label { .*? }/, ""));
        } else {
            for (var i = 0; i < 300; ++i) {
                line = line.replace(/{\\rm/, "\\mathrm{");
                line = line.replace(/{ \\rm/, "\\mathrm{");
                line = line.replace(/\\rm{/, "\\mathrm{");

                line = line.replace(/{\\bf/, "\\mathbf{");
                line = line.replace(/{ \\bf/, "\\mathbf{");
                line = line.replace(/\\bf{/, "\\mathbf{");

                line = line.replace(/{\\sf/, "\\mathsf{");
                line = line.replace(/{ \\sf/, "\\mathsf{");
                line = line.replace(/\\sf{/, "\\mathsf{");

                line = line.replace(/{\\it/, "\\mathit{");
                line = line.replace(/{ \\it/, "\\mathit{");
                line = line.replace(/\\it{/, "\\mathit{");

                line = line.replace(/{\\frak/, "\\mathfrak{");
                line = line.replace(/{ \\frak/, "\\mathfrak{");
                line = line.replace(/\\frak{/, "\\mathfrak{");

                line = line.replace(/{\\tt/, "\\mathtt{");
                line = line.replace(/{ \\tt/, "\\mathtt{");
                line = line.replace(/\\tt{/, "\\mathtt{");

                line = line.replace(/{\\cal/, "\\mathcal{");
                line = line.replace(/{ \\cal/, "\\mathcal{");
                line = line.replace(/\\cal{/, "\\mathcal{");

                line = line.replace(/{\\Bbb/, "\\mathbb{");
                line = line.replace(/{ \\Bbb/, "\\mathbb{");
                line = line.replace(/\\Bbb{/, "\\mathbb{");
            }

            // console.log(line)

            var tree = katex.__parse(line, {}); //START FROM HERE
            // console.log('Norm str', norm_str)

            buildExpression(tree, new options({}));
            for (var i = 0; i < 300; ++i) {
                norm_str = norm_str.replace('SSSSSS', '$');
                norm_str = norm_str.replace(' S S S S S S', '$');
            }
            console.log(norm_str.replace(/\\label { .*? }/, ""));
        }
    } catch (e) {
        console.error(line);
        console.error(norm_str);
        console.error(e);
        console.log();
    }
    global_str = ""
    norm_str = ""
})



// This is a LaTeX AST to LaTeX Renderer (modified version of KaTeX AST-> MathML).
norm_str = ""

var groupTypes = {};

groupTypes.mathord = function(group, options) {
    const rm_group = [
        "\\medspace", "\\thickspace", "\\hfill",
        "\\negthinspace", "\\negmedspace",
        "\\negthickspace", "\\mathstrut", "\\kern", "\\mkern"
    ]
    if (rm_group.includes(group.value)) {
        norm_str = norm_str + "\\, ";
    }
    else if (group.value == "\\lparen") {
        norm_str = norm_str + "( ";
    }
    else if (group.value == "\\rparen") {
        norm_str = norm_str + ") ";
    }
    else if (group.value == "\\lang") {
        norm_str = norm_str + "\\langle ";
    }
    else if (group.value == "\\rang") {
        norm_str = norm_str + "\\rangle ";
    }
    else {
        if (options.font == "mathrm" && !group.value.startsWith("\\")){
            for (i = 0; i < group.value.length; ++i ) {
                if (group.value[i] == " ") {
                    norm_str = norm_str + group.value[i] + "\, ";
                } else {
                    norm_str = norm_str + group.value[i] + " ";
                }
            }
        }
        else if (group.value == "\\textsf") {
            norm_str = norm_str + "\\mathsf" + " ";
        }
        else if (group.value == "\\textbf") {
            norm_str = norm_str + "\\mathbf" + " ";
        }
        else if (group.value == "\\textit") {
            norm_str = norm_str + "\\mathit" + " ";
        }
        else if (group.value == "\\texttt") {
            norm_str = norm_str + "\\mathtt" + " ";
        }
        else {
            norm_str = norm_str + group.value + " ";
        }
    }
};


groupTypes.textord = function(group, options) {
    if (group.value == "\\vert") {
        value = "|";
    }
    else if (group.value == "\\Vert") {
        value = "\\|";
    }
    else {
        value = group.value
    }
    norm_str = norm_str + value + " ";
};

groupTypes.bin = function(group) {
    norm_str = norm_str + group.value + " ";
};


groupTypes.rel = function(group) {
    if (group.value == "\\thinspace") {
        norm_str = norm_str + "\\, ";
    }
    else if (group.value == "\\hskip" || group.value == "\\mskip") {
        norm_str = norm_str + "\\, ";
    }
    else if (group.value == "\\gt") {
        norm_str = norm_str + "> ";
    }
    else if (group.value == "\\lt") {
        norm_str = norm_str + "< ";
    }

    else {
        norm_str = norm_str + group.value + " ";
    }
};


groupTypes.open = function(group) {
    if (group.value == "\\lbrack") {
        value = "[";
    }
    else if (group.value == "\\lbrace") {
        value = "\\{";
    }
    else if (group.value == "\\lvert") {
        value = "|";
    }
    else if (group.value == "\\lVert") {
        value = "\\|";
    }
    else {
        value = group.value;
    }
    norm_str = norm_str + value + " ";
};

groupTypes.close = function(group) {
    if (group.value == "\\rbrack") {
        value = "]";
    }
    else if (group.value == "\\rbrace") {
        value = "\\}";
    }
    else if (group.value == "\\rvert") {
        value = "|";
    }
    else if (group.value == "\\rVert") {
        value = "\\|";
    }
    else {
        value = group.value;
    }
    norm_str = norm_str + value + " ";
};


groupTypes.inner = function(group) {
    norm_str = norm_str + group.value + " ";
};


groupTypes.punct = function(group) {
    norm_str = norm_str + group.value + " ";
};


groupTypes.ordgroup = function(group, options) {
    norm_str = norm_str + "{ ";

    buildExpression(group.value, options);

    norm_str = norm_str +  "} ";
};


groupTypes.text = function(group, options) {

    norm_str = norm_str + "\\mathrm { ";

    buildExpression(group.value.body, options);
    norm_str = norm_str + "} ";
};


groupTypes.color = function(group, options) {
    var inner = buildExpression(group.value.value, options);

    var node = new mathMLTree.MathNode("mstyle", inner);

    node.setAttribute("mathcolor", group.value.color);

    return node;
};

groupTypes.supsub = function(group, options) {
    buildGroup(group.value.base, options);

    if (group.value.sub) {
        norm_str = norm_str + "_ ";
        if (group.value.sub.type != 'ordgroup') {
            norm_str = norm_str + "{ ";
            buildGroup(group.value.sub, options);
            norm_str = norm_str + "} ";
        } else {
            buildGroup(group.value.sub, options);
        }

    }

    if (group.value.sup) {
        norm_str = norm_str + "^ ";
        if (group.value.sup.type != 'ordgroup') {
            norm_str = norm_str + "{ ";
            buildGroup(group.value.sup, options);
            norm_str = norm_str + "} ";
        } else {
            buildGroup(group.value.sup, options);
        }
    }

};

groupTypes.genfrac = function(group, options) {
    if (!group.value.hasBarLine) {
        norm_str = norm_str + "\\binom ";
    } else {
        norm_str = norm_str + "\\frac ";
    }
    buildGroup(group.value.numer, options);
    buildGroup(group.value.denom, options);
};

groupTypes.array = function(group, options) {
    norm_str = norm_str + "\\begin{array} { ";
    if (group.value.cols) {
        group.value.cols.map(function(start) {
            if (start && start.align) {
                norm_str = norm_str + start.align + " ";}});
    } else {
        group.value.body[0].map(function(start) {
            norm_str = norm_str + "l ";
        } );
    }
    norm_str = norm_str + "} ";
    group.value.body.map(function(row) {
        if (row[0].value.length > 0) {
            out = row.map(function(cell) {
                buildGroup(cell, options);
                norm_str = norm_str + "& ";
            });
            norm_str = norm_str.substring(0, norm_str.length-2) + "\\\\ ";
        }
    });
    norm_str = norm_str + "\\end{array} ";
};

groupTypes.sqrt = function(group, options) {
    var node;
    if (group.value.index) {
        norm_str = norm_str + "\\sqrt [ ";
        buildExpression(group.value.index.value, options);
        norm_str = norm_str + "] ";
        buildGroup(group.value.body, options);
    } else {
        norm_str = norm_str + "\\sqrt ";
        buildGroup(group.value.body, options);
    }
};

groupTypes.leftright = function(group, options) {
    norm_str = norm_str + "\\left" + group.value.left + " ";
    buildExpression(group.value.body, options);
    norm_str = norm_str + "\\right" + group.value.right + " ";
};

groupTypes.accent = function(group, options) {
    if (group.value.base.type != 'ordgroup') {
        norm_str = norm_str + group.value.accent + " { ";
        buildGroup(group.value.base, options);
        norm_str = norm_str + "} ";
    } else {
        norm_str = norm_str + group.value.accent + " ";
        buildGroup(group.value.base, options);
    }
};

groupTypes.spacing = function(group) {
    var node;
    if (
        group.value == " " | group.value == "\\quad" |
        group.value == "~" | group.value == "\\!" |
        group.value == "\\:"| group.value == "\\;" |
        group.value == "\\qquad" | group.value == "\\enspace" |
        group.value == "\\ "
    ) {
        norm_str = norm_str + "\\, ";
    } else {
        norm_str = norm_str + group.value + " ";
    }
    return node;
};

groupTypes.op = function(group) {
    var node;

    // TODO(emily): handle big operators using the `largeop` attribute


    if (group.value.symbol) {
        // This is a symbol. Just add the symbol.
        norm_str = norm_str + group.value.body + " ";

    } else {
        if (group.value.limits == false) {
            norm_str = norm_str + "\\\operatorname { ";
        } else {
            norm_str = norm_str + "\\\operatorname* { ";
        }
        for (i = 1; i < group.value.body.length; ++i ) {
            norm_str = norm_str + group.value.body[i] + " ";
        }
        norm_str = norm_str + "} ";
    }
};

groupTypes.katex = function(group) {
    var node = new mathMLTree.MathNode(
        "mtext", [new mathMLTree.TextNode("KaTeX")]);

    return node;
};


groupTypes.font = function(group, options) {
    var font = group.value.font;
    const space = ["hspace", "vspace"]
    if (!space.includes(font)) {
        if (font == "mbox" || font == "hbox" || font=="fbox" || font=="framebox"
            || font == "makebox"
        ) {
            font = "mathrm";
        }
        else if (font == "bm") {
            font = "boldsymbol"
        }
        norm_str = norm_str + "\\" + font + " ";
        buildGroup(group.value.body, options.withFont(font));
    }
    else {
        norm_str = norm_str + "\\, ";
    }
};


groupTypes.delimsizing = function(group) {
    var children = [];
    norm_str = norm_str + group.value.funcName + " " + group.value.value + " ";
};


groupTypes.styling = function(group, options) {
    if (group.value.original == "\\textstyle" ) {
        value = "\\displaystyle";
    }
    else {
        value = group.value.original;
    }
    norm_str = norm_str + " " + value + " ";
    buildExpression(group.value.value, options);

};


groupTypes.sizing = function(group, options) {

    if (group.value.original == "\\rm" || group.value.original == "\\textrm") {
        norm_str = norm_str + "\\mathrm { ";
        buildExpression(group.value.value, options.withFont("mathrm"));
        norm_str = norm_str + "} ";
    }
    else if (group.value.original == "\\bf"|| group.value.original == "\\textbf") {
        norm_str = norm_str + "\\mathbf { ";
        buildExpression(group.value.value, options.withFont("mathbf"));
        norm_str = norm_str + "} ";
    }
    else if (group.value.original == "\\sf"|| group.value.original == "\\textsf") {
        norm_str = norm_str + "\\mathsf { ";
        buildExpression(group.value.value, options.withFont("mathsf"));
        norm_str = norm_str + "} ";
    }
    else if (group.value.original == "\\it"|| group.value.original == "\\textit") {
        norm_str = norm_str + "\\mathit { ";
        buildExpression(group.value.value, options.withFont("mathit"));
        norm_str = norm_str + "} ";
    }
    else if (group.value.original == "\\frak") {
        norm_str = norm_str + "\\mathfrak { ";
        buildExpression(group.value.value, options.withFont("mathfrak"));
        norm_str = norm_str + "} ";
    }
    else if (group.value.original == "\\tt") {
        norm_str = norm_str + "\\mathtt { ";
        buildExpression(group.value.value, options.withFont("mathtt"));
        norm_str = norm_str + "} ";
    }
    else if (group.value.original == "\\Bbb") {
        norm_str = norm_str + "\\mathbb { ";
        buildExpression(group.value.value, options.withFont("mathbb"));
        norm_str = norm_str + "} ";
    }
    else if (group.value.original == "\\cal") {
        norm_str = norm_str + "\\mathcal { ";
        buildExpression(group.value.value, options.withFont("mathcal"));
        norm_str = norm_str + "} ";
    }
    else {
        norm_str = norm_str + " " + group.value.original + " ";
        buildExpression(group.value.value, options);
    }
};


groupTypes.overline = function(group, options) {
    norm_str = norm_str + "\\overline { ";

    buildGroup(group.value.body, options);
    norm_str = norm_str + "} ";
    norm_str = norm_str;

};


groupTypes.underline = function(group, options) {
    norm_str = norm_str + "\\underline { ";
    buildGroup(group.value.body, options);
    norm_str = norm_str + "} ";
    norm_str = norm_str;

};


// groupTypes.rule = function(group) {
//     norm_str = norm_str + "\\rule { "+group.value.width.number+" "+group.value.width.unit+" } { "+group.value.height.number+" "+group.value.height.unit+ " } ";

// };


groupTypes.rule = function(group) {
    norm_str = norm_str + " ";
};


groupTypes.llap = function(group, options) {
    norm_str = norm_str + " ";
};


// groupTypes.llap = function(group, options) {
//     norm_str = norm_str + "\\llap ";
//     buildGroup(group.value.body, options);
// };


// groupTypes.rlap = function(group, options) {
//     norm_str = norm_str + "\\rlap ";
//     buildGroup(group.value.body, options);

// };

groupTypes.rlap = function(group, options) {
    norm_str = norm_str + " ";
};

groupTypes.phantom = function(group, options, prev) {
    // console.log('Phantom value', group.value.value)
    // buildExpression(group.value.value, options);
    // DO nothing
    norm_str = norm_str + " "
};


/**
 * Takes a list of nodes, builds them, and returns a list of the generated
 * MathML nodes. A little simpler than the HTML version because we don't do any
 * previous-node handling.
 */
var buildExpression = function(expression, options) {
    var groups = [];
    for (var i = 0; i < expression.length; i++) {
        var group = expression[i];
        // console.log('group name', group)
        buildGroup(group, options);
    }
    // console.log(norm_str);
    // return groups;
};


/**
 * Takes a group from the parser and calls the appropriate groupTypes function
 * on it to produce a MathML node.
 */
var buildGroup = function(group, options) {
    if (groupTypes[group.type]) {
        groupTypes[group.type](group, options);
    } else {
        throw new ParseError(
            "Got group of unknown type: '" + group.type + "'");
    }
};
