Papers and reviews in this section can be obtained from the NIPS website by running the commands below at the root directory of this repository.
More details on the science-parse library can be found [here](https://github.com/allenai/science-parse/).

```shell
#!/usr/bin/env bash

From the code directory, run 

python ./data_prepare/crawler/NIPS_crawl.py <year> <output directory> <no_pdf (optional)>

Where:

year -- NIPS year to download

output directory -- output directory to save data to

no_pdf -- don't download pdfs, but just reviews


To download all files, run the following:

for year in {2013..2017}; do
	python ./data_prepare/crawler/NIPS_crawl.py $year ../data/nips_2013-2017/$year
    mkdir -p data/nips_2013-2017/$year/parsed_pdfs/
    for pdf_filename in $( ls data/nips_2013-2017/$year/pdfs/*.pdf ); do
      java -Xmx6g -jar lib/science-parse-cli-assembly-1.2.9-SNAPSHOT.jar $i > data/nips_2013-2017/$year/parsed_pdfs/$(basename ${pdf_filename}).json
    done
done 
```
