#!/bin/bash

# --- install_isabelle.sh ---
# Script to automatically install dependencies: Isabelle and AFP
# If you agree that the operations below are safe, 
# make this script executable: chmod +x install_isabelle.sh
# and run it: ./install_isabelle.sh
# If you want to use your own configuration, make sure that this project ./src/main/scala/directories.scala
# points to the correct locations of Isabelle and AFP

echo "Creating dependencies directory..."
mkdir lib                                                    # create directory of dependencies
cd lib                                                       # go to directory of dependencies

echo "Downloading and extracting Isabelle 2025..."
curl -sO https://isabelle.in.tum.de/dist/Isabelle2025-1_linux.tar.gz  # downloading
tar -xzf Isabelle2025-1_linux.tar.gz                                  # extracting
rm Isabelle2025-1_linux.tar.gz                                        # removing compressed version

echo "Downloading and extracting AFP..."
curl -sO https://www.isa-afp.org/release/afp-current.tar.gz         # downloading
tar -xzf afp-current.tar.gz                                         # extracting
rm afp-current.tar.gz                                               # removing compressed version
mv afp* afp                                                         # renaming the extraction

echo "Making Isabelle aware of the AFP..."
ISABELLE="./Isabelle2025-1/bin/isabelle"                              # path to isabelle
$ISABELLE components -u "./afp/thys/"