# -*- coding: utf-8 -*-
import re


file_path = "../README.md"
with open(file_path, "r") as f:
    content = f.readlines()

for c in content:
    c = c.strip()
    pattern = r"^#+\s[0-9.]+\s"
    r = re.match(pattern, c)
    if r:
        c1 = re.sub(pattern, "", c)
        c2 = re.sub(r"#+\s", "", c)
        string = '<a href="#{}">{}</a>  '.format(c1, c2)
        print(string)

