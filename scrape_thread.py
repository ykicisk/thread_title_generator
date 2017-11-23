#/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
import time
import argparse
from urllib import request
from urllib.error import HTTPError, URLError
from socket import timeout
from lxml import html


def urlopen_with_retry(url: str, n_retry: int, interval: float):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0",
    }
    err, res = None, None
    for i in range(n_retry):
        try:
            req = request.Request(url=url, headers=headers)
            res = request.urlopen(req, timeout=10)
            break
        except (HTTPError, URLError) as e:
            print(e.code, e.msg)
            if e.code not in [403, 503]:
                raise
            err = e
            time.sleep(interval*10)
    if res is None and err is not None:
        raise err
    return res


def scrape_thread(dst_dir: str, start_num: int, end_num: int, interval: float):
    url_fmt="https://hawk.5ch.net/livejupiter/kako/kako{0:04d}.html"

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for idx in range(start_num, end_num+1):
        url = url_fmt.format(idx)
        print("open:", url)

        for i in range(3):
            try:
                res = urlopen_with_retry(url, n_retry=3, interval=interval)
                data = res.read()
                break
            except timeout:
                print("timeout...")
                time.sleep(interval*10)
        tree = html.fromstring(data)

        titles_odd = tree.xpath('//div[@class="main"]/p[@class="main_odd"]/span[@class="title"]/a')
        titles_even = tree.xpath('//div[@class="main"]/p[@class="main_even"]/span[@class="title"]/a')
        lines_odd = tree.xpath('//div[@class="main"]/p[@class="main_odd"]/span[@class="lines"]')
        lines_even = tree.xpath('//div[@class="main"]/p[@class="main_even"]/span[@class="lines"]')

        dst_path = os.path.join(dst_dir, "{0:04d}.tsv".format(idx))

        print("save as:", dst_path)
        with open(dst_path, "w") as f:
            for t, l in zip(titles_odd, lines_odd):
                f.write("{0}\t{1}\n".format(t.text, l.text))
            for t, l in zip(titles_even, lines_even):
                f.write("{0}\t{1}\n".format(t.text, l.text))

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dst", help="output path")
    parser.add_argument("--start", type=int, default=0,
                        help="start pagenum (include)")
    parser.add_argument("--end", type=int, default=2286,
                        help="end pagenum (include)")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="access interval to 5ch")
    args = parser.parse_args()
    scrape_thread(args.dst, args.start, args.end, args.interval)


if __name__ == "__main__":
    main()
