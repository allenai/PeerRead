Papers and reviews in this section can be obtained from the NIPS website by running the commands below at the root directory of this repository.
More details on the science-parse library can be found [here](https://github.com/allenai/science-parse/).

```
#!/usr/bin/env bash

python code/download-nips-2013-2016.py
for split in train dev test; do
  mkdir data/nips_2013-2016/$split/parsed_pdfs/
  for pdf_filename in $( ls data/nips_2013-2016/$split/pdfs/*.pdf ); do
    java -Xmx6g -jar lib/science-parse-cli-assembly-1.2.9-SNAPSHOT.jar $i > data/nips_2013-2016/$split/parsed_pdfs/$(basename ${pdf_filename}).json
  done
done
```
