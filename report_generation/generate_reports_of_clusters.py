import os
import re
reports_path = "/home/farhood/Projects/Cognitive-project/Final/cognitive_package/res/reports/"

source_prefix = "kmeans_report_"
regex_pattern = r'key phrase: [a-zA-Z\s]'
string = "key_phrase: cognitive linguistics study language                         score: 0.058985 cluster: 1"

print(re.search(regex_pattern, string))
# for root, __, files in os.walk(reports_path):
#     for file_name in files: 
#         report_type = file_name[:-4].replace(source_prefix,"")
#         with open(os.path.join(reports_path, file_name)) as myfile:
#             for line in myfile.readlines():
                
