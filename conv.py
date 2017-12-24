import sys

import xml.etree.ElementTree as ET

tree = ET.parse(sys.argv[1])

print(",".join([sys.argv[1].split('_')[1],
                tree.find('COOK')[1].text,
                tree.find('FOOD1')[1].text,
                tree.find('OUTPUT_PERCENT').text,
                tree.find('FOOD2')[1].text]))
