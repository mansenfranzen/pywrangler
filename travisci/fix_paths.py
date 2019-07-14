"""This module fixes paths of the coverage.xml manually since codecov is
unable to correctly read the coverage report generated within tox environments.

More specifically, it replaces the tox file paths with `src` file paths which
is expected by codecov.

"""

import re

REGEX_FILENAME = r'(.*)\.tox.*site-packages(.*)'
regex_filename = re.compile(REGEX_FILENAME)

with open("coverage.xml", "r") as infile:
    print("Reading original coverage.xml report.")
    replaced  = [regex_filename.sub("\g<1>src\g<2>", line)
                 for line in infile]

with open("coverage.xml", "w") as outfile:
    print("Overwriting coverage.xml. Replaced {} paths.".format(len(replaced)))
    outfile.writelines(replaced)
