# mostly taken from http://code.google.com/p/latexmath2png/
# install preview.sty
# Modify by duyla4
import os
import re
import sys
import io
import glob
import tempfile
import shlex
import subprocess
import traceback
from PIL import Image
from threading import Timer


class Latex:
    BASE = r"""
\documentclass[varwidth]{standalone}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage[active,tightpage,displaymath,textmath]{preview}
\begin{document}
\thispagestyle{empty}
\newlength{\mylength}
%s
\end{document}
"""

    CHECK_LENGTH = r"""\begin{displaymath}
\settowidth{\mylength}{$\displaystyle %s$}
\ifdim\mylength>\linewidth
    \resizebox{\linewidth}{!}{$\displaystyle %s$}
\else
        %s
\fi
\end{displaymath}"""

    TIMEOUT = 20

    def __init__(self, math, dpi=250):
        """takes list of math code. `returns each element as PNG with DPI=`dpi`"""
        # assert isinstance(math, dict)
        self.math = math
        self.dpi = dpi
        self.prefix_line = self.BASE.split("\n").index(
            "%s"
        )  # used for calculate error formula index
        block_length = len(self.CHECK_LENGTH.split("\n"))
        self.block_segment = [
            [idx * block_length, (idx + 1) * block_length - 1]
            for idx, _ in enumerate(self.math)
        ]
        self.kill_proc = lambda p: p.kill()

    def write(self, return_bytes=False):
        try:
            workdir = tempfile.gettempdir()
            fd, texfile = tempfile.mkstemp(".tex", "eq", workdir, True)

            with os.fdopen(fd, "w+") as f:
                document = ""
                for math in self.math:
                    document += self.CHECK_LENGTH % (math, math, math)
                    document += "\n"
                document = document.strip()
                document = self.BASE % document
                f.write(document)

            # with open('test_sample.txt', 'w') as f:
            #     document = ''
            #     for math in self.math:
            #         document += self.CHECK_LENGTH%(math, math, math)
            #         document += '\n'
            #     document = document.strip()
            #     document = self.BASE%document
            #     f.write(document)

            png, error_index = self.convert_file(
                texfile, workdir, return_bytes=return_bytes
            )

            return png, error_index

        finally:
            if os.path.exists(texfile):
                try:
                    os.remove(texfile)
                except PermissionError:
                    pass

    def convert_file(self, infile, workdir, return_bytes=False):
        infile = infile.replace("\\", "/")
        try:
            # Generate the PDF file
            #  not stop on error line, but return error line index,index start from 1
            cmd = (
                "pdflatex -interaction nonstopmode -file-line-error -output-directory %s %s"
                % (workdir.replace("\\", "/"), infile)
            )

            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            timer = Timer(self.TIMEOUT, self.kill_proc, [p])
            try:
                timer.start()
                sout, serr = p.communicate()
            finally:
                timer.cancel()

            # extract error line from sout
            error_index, _ = extract(
                text=sout, expression=r"%s:(\d+)" % os.path.basename(infile)
            )

            # extract success rendered equation
            block_error_idx = []
            if error_index != []:
                error_index = list(set(error_index))
                for error_idx in error_index:
                    error_idx = int(error_idx) - self.prefix_line - 1
                    for block_idx, block in enumerate(self.block_segment):
                        if error_idx in list(range(block[0], block[1] + 1)):
                            block_error_idx.append(block_idx)
                            break

            # Convert the PDF file to PNG's
            pdffile = infile.replace(".tex", ".pdf")

            result, _ = extract(
                text=sout, expression="Output written on %s \((\d+)? page" % pdffile
            )

            if not len(result) or (int(result[0]) != len(self.math)):
                raise Exception(
                    "pdflatex rendering error, generated %d formula's page, but the total number of formulas is %d."
                    % (0 if not len(result) else int(result[0]), len(self.math))
                )

            pngfile = os.path.join(workdir, infile.replace(".tex", ".png"))

            cmd = "convert -density %i -colorspace gray %s -quality 100 %s" % (
                self.dpi,
                pdffile,
                pngfile,
            )  # -bg Transparent -z 9

            cmd = "magick " + cmd if sys.platform == "win32" else cmd

            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            timer = Timer(self.TIMEOUT, self.kill_proc, [p])
            try:
                timer.start()
                sout, serr = p.communicate()
            finally:
                timer.cancel()

            if p.returncode != 0:
                raise Exception(
                    "PDFpng error",
                    serr,
                    cmd,
                    os.path.exists(pdffile),
                    os.path.exists(infile),
                )

            if return_bytes:
                if len(self.math) > 1:
                    png = [
                        open(pngfile.replace(".png", "") + "-%i.png" % i, "rb").read()
                        for i in range(len(self.math))
                    ]
                else:
                    png = [open(pngfile.replace(".png", "") + ".png", "rb").read()]
            else:
                # return path
                if len(self.math) > 1:
                    png = [
                        (pngfile.replace(".png", "") + "-%i.png" % i)
                        for i in range(len(self.math))
                    ]
                else:
                    png = [(pngfile.replace(".png", "") + ".png")]

            return png, block_error_idx

        except Exception as e:
            print("Error rendering", e)

        finally:
            # Cleanup temporaries
            basefile = infile.replace(".tex", "")
            tempext = [".aux", ".pdf", ".log"]
            if return_bytes:
                ims = glob.glob(basefile + "*.png")
                for im in ims:
                    os.remove(im)

            for te in tempext:
                tempfile = basefile + te
                if os.path.exists(tempfile):
                    os.remove(tempfile)


__cache = {}


def tex2png(eq, **kwargs):
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]


def tex2pil(tex, return_error_index=False, **kwargs):
    pngs, error_index = Latex(tex, **kwargs).write(return_bytes=True)
    images = [Image.open(io.BytesIO(d)) for d in pngs]
    return (images, error_index) if return_error_index else images


def extract(text, expression=None):
    """extract text from text by regular expression

    Args:
        text (str): input text
        expression (str, optional): regular expression. Defaults to None.

    Returns:
        str: extracted text
    """
    try:
        pattern = re.compile(expression)
        results = re.findall(pattern, text)
        return results, True if len(results) != 0 else False
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = r"\begin{equation}\mathcal{ L}\nonumber\end{equation}"

    print("Equation is: %s" % src)
    print(Latex(src).write())
