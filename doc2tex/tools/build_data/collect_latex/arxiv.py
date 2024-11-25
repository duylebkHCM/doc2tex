# modified from https://github.com/soskek/arxiv_leaks

import argparse
import subprocess
import os
import glob
import re
import gc
import random
import argparse
import logging
import tarfile
import tempfile
import logging
import requests
import urllib.request
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import numpy as np
from urllib.error import HTTPError
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Manager
from tools.build_data.latex_processing.extract_latex import find_math
from tools.build_data.collect_latex.scraping import recursive_search
from tools.build_data.collect_latex.demacro import *
from bs4 import BeautifulSoup
from lxml import html
from functools import partial

logging.getLogger().setLevel(logging.INFO)
arxiv_id = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?!\d)|" r"(?<!\d)(\d{7})(?!\d)")
arxiv_base = "https://export.arxiv.org/e-print/"
date_format = None


def get_num_papers(url):
    """
    //*[@id="dlpage"]/small[1]/text()[1]
    """
    try:
        page = requests.get(url)
    except HTTPError:
        print("Cannot access this url")
        return None
    else:
        try:
            soup = BeautifulSoup(page.content, "html.parser")
            dom = html.fromstring(bytes(str(soup), encoding="utf-8"))
            num_papers = dom.xpath('//*[@id="dlpage"]/small[1]/text()[1]')[0]
            num_papers = [int(word) for word in num_papers.split() if word.isdigit()]
            assert len(num_papers) == 1
            return num_papers[0]
        except IndexError:
            return None


def get_year_publish(url):
    """
    //*[@id="content"]/ul/li[4]
    """
    try:
        page = requests.get(url)
    except HTTPError:
        print("Cannot access this url")
        return None
    else:
        soup = BeautifulSoup(page.content, "html.parser")
        dom = html.fromstring(bytes(str(soup), encoding="utf-8"))
        year_infos = dom.xpath('//*[@id="content"]/ul/li[4]')[0]

        year_lst = []
        for child in year_infos.iter("a"):
            year_lst.append(child.text)

        year_lst = [
            str(int(yr[-2:])).zfill(2) if int(yr[-2:]) < 10 else str(int(yr[-2:]))
            for yr in year_lst
        ]

        return year_lst


def get_all_arxiv_ids(text, skip: List[str] = []):
    """returns all arxiv ids present in a string `text`"""
    ids = []
    for id in arxiv_id.findall(text):
        if isinstance(id, tuple) and len(id) > 1:
            id = [item for item in id if len(item) > 0][0]
        if skip is not None and id in skip:
            continue
        ids.append(id)
    return list(set(ids))


def download(url, dir_path="./"):
    idx = os.path.split(url)[-1]
    file_name = idx + ".tar.gz"
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        return file_path
    logging.info("\tdownload {}".format(url) + "\n")
    try:
        r = urllib.request.urlretrieve(url, file_path)
        return r[0]
    except (
        HTTPError,
        urllib.error.URLError,
        requests.exceptions.SSLError,
        ConnectionResetError,
    ) as e:
        logging.info("Could not download %s" % url)
        logging.info("Error: ", e)
        return 0


def read_tex_files(file_path: str, demacro: bool = False) -> str:
    """Read all tex files in the latex source at `file_path`. If it is not a `tar.gz` file try to read it as text file.

    Args:
        file_path (str): Path to latex source
        demacro (bool, optional): Deprecated. Call external `de-macro` program. Defaults to False.

    Returns:
        str: All Latex files concatenated into one string.
    """
    tex = ""
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                tf = tarfile.open(file_path, "r")
                tf.extractall(tempdir)
                tf.close()
                texfiles = [
                    os.path.abspath(x)
                    for x in glob.glob(
                        os.path.join(tempdir, "**", "*.tex"), recursive=True
                    )
                ]
            except tarfile.ReadError as e:
                texfiles = [file_path]  # [os.path.join(tempdir, file_path+'.tex')]
            except EOFError as e:
                texfiles = [file_path]
            if demacro:
                ret = subprocess.run(
                    ["de-macro", *texfiles], cwd=tempdir, capture_output=True
                )
                if ret.returncode == 0:
                    texfiles = glob.glob(
                        os.path.join(tempdir, "**", "*-clean.tex"), recursive=True
                    )
            for texfile in texfiles:
                try:
                    ct = open(texfile, "r", encoding="utf-8").read()
                    tex += ct
                except UnicodeDecodeError as e:
                    # logging.debug(e)
                    pass
    except Exception as e:
        pass

    tex = pydemacro(tex)
    return tex


def download_paper(arxiv_id, dir_path="./"):
    url = arxiv_base + arxiv_id
    return download(url, dir_path)


def read_paper(targz_path, delete=False, demacro=False):
    paper = ""
    if targz_path != 0:
        paper = read_tex_files(targz_path, demacro=demacro)
        if delete:
            os.remove(targz_path)
    return paper


def parse_arxiv(save, math=None, demacro=True, id=None):
    if save is None:
        dir = tempfile.gettempdir()
    else:
        dir = save

    try:
        text = read_paper(download_paper(id, dir), delete=save is None, demacro=demacro)
        if text is None:
            raise ValueError
        text = find_math(text)
        if math is None:
            return text
        else:
            math.put(text)
    except ValueError:
        pass


# initialize the worker process
def init_worker():
    # get the pid for the current worker process
    pid = os.getpid()
    print(f"Worker PID: {pid}", flush=True)


def write_math2disk(visited, math=None):
    for l, name in zip([visited, math], ["visited_arxiv.txt", "math_arxiv.txt"]):
        if l is None:
            continue

        f = os.path.join(args.out, name)

        if not os.path.exists(f):
            open(f, "w").write("")

        f = open(f, "a", encoding="utf-8")
        for element in l:
            f.write(element)
            f.write("\n")

        f.close()


def write_math2disk_multiproces(math):
    f = os.path.join(args.out, "math_arxiv.txt")
    if not os.path.exists(f):
        open(f, "w").write("")
    with open(f, "a", encoding="utf-8") as f_i:
        while True:
            res = math.get()
            if res == "kill":
                break
            for element in res:
                f_i.write(element)
                f_i.write("\n")


def retrieve_math(process_ids, process_urls):
    total_math = []
    for id in tqdm(process_ids):
        math = parse_arxiv(save=args.save, demacro=args.demacro, id=id)
        total_math.extend(math)

    write_math2disk(process_urls, math)


def retrieve_math_multiprocess(process_ids):
    logging.info("Creating pool with %d threads" % args.num_threads)

    with Manager() as manager:
        pool = ThreadPool(args.num_threads, initializer=init_worker)

        math = manager.Queue()
        _ = pool.apply_async(write_math2disk_multiproces, (math,))

        func = partial(parse_arxiv, args.save, math, args.demacro)
        res = pool.map_async(func, process_ids)
        res.get()

        math.put("kill")

        pool.close()
        pool.join()
        gc.collect()


def get_try_session():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def process_args():
    parser = argparse.ArgumentParser(description="Extract math from arxiv")
    parser.add_argument(
        "-m",
        "--mode",
        default="top100",
        choices=["top", "ids", "dirs"],
        help="Where to extract code from. top: current 100 arxiv papers (-m top int for any other number of papers), id: specific arxiv ids. \
                              Usage: `python arxiv.py -m ids id001 [id002 ...]`, dirs: a folder full of .tar.gz files. Usage: `python arxiv.py -m dirs directory [dir2 ...]`",
    )
    parser.add_argument(nargs="*", dest="args", default=[])
    parser.add_argument(
        "-l",
        "--url_path",
        default="tools/build_data/collect_latex/research_field_urls.txt",
        help="List of url to download arxiv paper using top mode",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=str(
            Path(__file__).resolve().absolute().parent.parent.joinpath("crawl_math")
        ),
        help="output directory",
    )
    parser.add_argument(
        "-d",
        "--demacro",
        dest="demacro",
        action="store_true",
        help="Deprecated - Use de-macro (Slows down extraction, may but improves quality). Install https://www.ctan.org/pkg/de-macro",
    )
    parser.add_argument(
        "-s",
        "--save",
        default=None,
        type=str,
        help="When downloading files from arxiv. Where to save the .tar.gz files. Default: Only temporary",
    )
    parser.add_argument(
        "--num-threads",
        dest="num_threads",
        type=int,
        default=4,
        help=("Number of threads, default=4."),
    )
    parser.add_argument(
        "-nl",
        "--num-limit",
        default=40,
        help="Set maximum number of paper to process for all process at a time to prevent from out-of-memory problem",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = process_args()

    if "." in args.out:
        args.out = os.path.dirname(args.out)
    skips = os.path.join(args.out, "visited_arxiv.txt")
    if os.path.exists(skips):
        skip = open(skips, "r", encoding="utf-8").read().split("\n")
    else:
        skip = []

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    try:
        if args.mode == "ids":
            visited, math = recursive_search(
                parse_arxiv,
                args.args,
                skip=skip,
                unit="paper",
                save=args.save,
                demacro=args.demacro,
            )
            write_math2disk(visited, math)

        elif args.mode == "top":
            arxiv_session = get_try_session()

            num = int(args.args[0])
            total_ids = []
            total_url = []
            url_list = open(str(args.url_path), "r", encoding="utf-8").readlines()
            random.shuffle(url_list)
            for url in url_list:
                url = url.strip()
                publish_year_lst = get_year_publish(url)
                if publish_year_lst is None:
                    continue
                random.shuffle(publish_year_lst)
                for year in publish_year_lst:
                    months = list(range(1, 13))
                    random.shuffle(months)
                    for month in months:
                        url = url.replace("archive", "list")
                        full_url = (
                            url
                            + year
                            + (str(month) if month > 9 else str(month).zfill(2))
                        )
                        num_ids = get_num_papers(full_url)
                        if num_ids is None:
                            continue
                        all_full_url = full_url + "/?show=%i" % num_ids
                        try:
                            full_url_info = arxiv_session.get(all_full_url)
                        except Exception as e:
                            logging.debug("Error ", e)
                            continue
                        ids = get_all_arxiv_ids(full_url_info.text, skip)
                        total_ids += ids
                        total_url += [full_url + "/" + id for id in ids if len(ids)]
                        if len(total_ids) > num:
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

            total_ids, total_url = total_ids[:num], total_url[:num]
            assert len(total_ids) >= args.num_threads

            process_infos = list(zip(total_ids, total_url))
            try:
                process_batches = np.array_split(
                    process_infos, int(len(process_infos) / args.num_limit)
                )
            except ValueError:
                process_batches = []

            if len(process_batches):
                for process_batch in process_batches:
                    process_ids, process_urls = list(zip(*process_batch.tolist()))
                    if args.num_threads > 1:
                        write_math2disk(process_urls)
                        retrieve_math_multiprocess(process_ids)
                    else:
                        retrieve_math(process_urls, process_ids)
            else:
                if args.num_threads > 1:
                    write_math2disk(total_url)
                    retrieve_math_multiprocess(total_ids)
                else:
                    retrieve_math(total_url, total_ids)

        elif args.mode == "dirs":
            files = []
            for folder in args.args:
                files.extend([os.path.join(folder, p) for p in os.listdir(folder)])
            math, visited = [], []
            for f in tqdm(files):
                try:
                    text = read_paper(f, delete=False, demacro=args.demacro)
                    math.extend(find_math(text))
                    visited.append(os.path.basename(f))
                except DemacroError as e:
                    logging.debug(f + str(e))
                    pass
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logging.debug(e)
                    raise e

            write_math2disk(visited, math)
        else:
            raise NotImplementedError
    except KeyboardInterrupt:
        pass
