
Papers and reviews in this section can be obtained from the NIPS website by running the command below at the root directory of this repository.

```shell
#!/usr/bin/env bash
cd code
python ./data_prepare/crawler/NIPS_crawl.py $year $output_directory (no_pdf)
```
Where `$year` is the four-digit NIPS year to download, `$output_directory` is the output directory where the files will be downloaded, and `no_pdf` is an optional argument for downloading the reviews only (and none of the pdf files which can be large). Once the PDF files have been downloaded, you can use the [science-parse](https://github.com/allenai/science-parse/) library to process them into json-formatted files.

For example, to download and process the years 2013-2017, run the following after installing science-parse at `./code/lib/`. The script will split them into train/dev/test and for each split parse them to pdfs/parsed_pdfs/reviews. Please be sure that pdfs/ and reviews/ should exist in advance for each year of NIPS papers.

```shell
#!/usr/bin/env bash
cd code
for year in {2013..2017}; do
    python ./data_prepare/crawler/NIPS_crawl.py $year ../data/nips_2013-2017/$year
    python ./data_prepare/prepare.py ../data/nips_2013-2017/&year
    done
done
```

In case for parsing all PDFs only, run the following code:

```shell
#!/usr/bin/env bash
cd code
for year in {2013..2017}; do
    python ./data_prepare/crawler/NIPS_crawl.py $year ../data/nips_2013-2017/$year
    mkdir -p data/nips_2013-2017/$year/parsed_pdfs/
    for pdf_filename in $( ls data/nips_2013-2017/$year/pdfs/*.pdf ); do
      java -Xmx6g -jar lib/science-parse-cli-assembly-1.2.9-SNAPSHOT.jar $i > data/nips_2013-2017/$year/parsed_pdfs/$(basename ${pdf_filename}).json
    done
done
```



