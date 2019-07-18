#!/usr/bin/python

from __future__ import print_function
import sys
import os
import re
import urllib3
import certifi

from models.Paper import *


def main(args):
    argc = len(args)
    no_pdf = None

    if argc < 3:
        print("Usage:",args[0],"<year> <output dir> <don't save pdfs -- optional>")
        return -1
    elif argc > 3:
        no_pdf = True

    year = args[1]
    out_dir = args[2]
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    number = str(int(year)-1987)
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

    url = "https://papers.nips.cc/book/advances-in-neural-information-processing-systems-"+number+"-"+year


    data = get_url(http, url)

    if data is None:
        return -2

    lines = str(data).split(r"\n")

    print("Got",len(lines),"lines")

    n_papers = 0
    for l in lines:
        if l.find('<li><a href="/paper') != -1:
            n_papers +=1

            data = handle_paper(l, out_dir, http, int(year), no_pdf)
            # break
            if data is None:
                print("#### Can't handle paper")
                continue

    print ("Found",n_papers,"papers")

def handle_paper(l, out_dir, http, year, no_pdf):
    l = re.sub("<var>([^\<]+)</var>", "\1", l)
    match = re.search('<li><a href="([^\"]+)">([^<]+)</a> (.*)</li>', l)

    if match is None:
        print("Bad line",l)
        return None

    authors_strs = match.group(3)[1:-1].split(">, <")
    title = match.group(2)

    authors = []
    for author_str in authors_strs:
        match2 = re.search('a [^>]+>([^<]+)</a', author_str)
        if match2 is None:
            print("No authors found for", title)
        else:
            authors.append(match2.group(1))

    data = handle_url2(match.group(1), out_dir, http, year, no_pdf)

    if data is None:
        return None

    [reviews, id, publication_type, abstract] = data

    conference = "NIPS"+str(year)
    p = Paper(title, abstract, id, reviews, authors, conference, True, None, publication_type)

    p.to_json(out_dir+"/reviews"+str(id)+".json")

    return True

def handle_url2(url, out_dir, http, year, no_pdf):
    url = "https://papers.nips.cc/"+url

    lines = get_url(http, url)

    if lines is None:
        return None

    lines = str(lines).split(r"\n")

    publication_type = None
    abstract = None
    reviews_url = None
    pdf_url = None

    for l in lines:
        # print("Check",l)
        if pdf_url is None:
            matched = re.search('<a href="([^"]+)">\[PDF\]</a>', l)

            if matched is not None:
                pdf_url = matched.group(1)
        elif reviews_url is None:
            matched = re.search('<a href="([^"]+)">\[Reviews\]</a>', l)

            if matched is not None:
                reviews_url = matched.group(1)
        elif publication_type is None:
            matched = re.search("<h3>Conference Event Type: ([a-zA-Z]+)</h3>", l)

            if matched is not None:
                publication_type = matched.group(1)
        elif abstract is None:
            # print("Testing",l)
            matched = re.search('<p class="abstract">(.*)</p>', l)

            if matched is not None:
                abstract = matched.group(1)
                break

    if abstract is None:
        print("Some fields missing in paper page:",url,"Other:",pdf_url, reviews_url, publication_type)
        abstract = ""
        
        if reviews_url is None:
            reviews_url = ""
            
        if publication_type is None:
            publication_type = ""
#        return None

    # print("PDF url:",pdf_url,"Review URL:",reviews_url,"type:",publication_type,"abstract:",abstract)

    reviews, id = handle_url3(http, reviews_url, year)

    if no_pdf is None and pdf_url is not None:
        pdf_url = "https://papers.nips.cc"+pdf_url

        pdf_data = get_url(http, pdf_url)

        if pdf_data is None:
            return None

        with open(out_dir+"/pdfs"+id+".pdf", 'wb') as ofh:
            ofh.write(pdf_data)

    return [reviews, id, publication_type, abstract]

def handle_url3(http, reviews_url, year):
    id = reviews_url.split("/")[-1].split(".")[0]
    reviews_url = "http:"+reviews_url
    confidence = None

    data = str(get_url(http, reviews_url))

    if data is None:
        return None, None

    reviews = []
    review = None
    in_div = 0
    if year < 2016:
        data = data.replace("\r\n", " ")
        data = re.sub(" +", " ", data)

        lines = re.split("<BR/?>", data, flags=re.IGNORECASE)


        for l in lines:
            # print(l)
            if re.search("<i>Submitted by Assigned_Reviewer_[0-9]+</i>", l, re.IGNORECASE):
                if review is not None:
                    r = Review(None, review)
                    reviews.append(r)

                review = ""
            elif review is not None:
                if not in_div:
                    match = re.search('<DIV class="?response"?>(.*)', l, re.IGNORECASE)

                    if match is not None:
                        review += match.group(1)
                        in_div = 1
                else:
                    match = re.search("(.*)</DIV>", l, re.IGNORECASE)

                    if match is not None:
                        review += match.group(1)
                        in_div = 0
                    else:
                        review += l
    else:
        lines = data.split("\n")
        has_confidence = 0
        for l in lines:
            if re.search("<h3>Reviewer [0-9]+</h3>", l, re.IGNORECASE):
                # print("Found reviewer - ",l)
                if review is not None:
                    r = Review(None, review, None, None, None, None, None, confidence)
                    reviews.append(r)

                review = ""
                confidence = None
            elif review is not None:
                if has_confidence:
                    match = re.search("<p>([0-9]+)\-.*</p>", l, re.IGNORECASE)

                    if match is None:
                        print("Line after confidence is not confidence line:",l)
                        return None
                    else:
                        confidence = match.group(1)
                        has_confidence = False

                elif re.search("[ \t]*<h4>Confidence in this Review</h4>", l, re.IGNORECASE) is not None:
                    has_confidence = True
                    continue
                else:
                    match = re.search("<p>(.*)</p>", l, re.IGNORECASE)

                    if match is not None:
                        review += match.group(1)
                        in_div = 1

    if review is not None:
        r = Review(None, review, None, None, None, None, None, confidence)
        reviews.append(r)

    if len(reviews) == 0:
        print("%%% no reviews found in",reviews_url,"!")
                            #re.findall("<div>.*</div>")
    return reviews, id


def get_url(http, url):
    try:
        response = http.request('GET', url)

        if response.status != 200:
            print("Problem, couldn't download", url,"(status is "+str(response.status)+")")
            return None
    except urllib3.exceptions.MaxRetryError:
        print("Max retries exceeded with url", url)
        return None

    return response.data



if __name__ == "__main__":
    sys.exit(main(sys.argv))
