#!/bin/bash

# --- install_scala_deps.sh ---
# Script to automatically install scala-isabelle
# Assumes that java is already installed
# If you agree that the operations below are safe, 
# make this script executable: chmod +x install_scala_deps.sh
# and run it: ./install_scala_deps.sh

chmod +x install_isabelle.sh
./install_isabelle.sh                                               # install Isabelle and AFP

echo "Downloading and installing scala-isabelle..."
git clone https://github.com/dominique-unruh/scala-isabelle.git     # cloning the repository
cd scala-isabelle                                                   # go to the directory

if command -v sbt &> /dev/null                                      # check if sbt is installed
then
    SBT="sbt"
else
    echo "Scala compiler 'sbt' not found. Installing local version..."
    
    curl -sL https://github.com/sbt/sbt/releases/download/v1.11.7/sbt-1.11.7.tgz | tar -xz -C "../"                                                               # download and extract sbt
    
    SBT="../sbt/bin/sbt"                                            # set SBT command

    if [ ! -x $SBT ]; then
        echo "ERROR: Failed to find or install local sbt launcher." >&2
        exit 1
    fi

    echo "Local 'sbt' installed to $SBT"
fi
echo "Compiling scala-isabelle with sbt..."
$SBT publishLocal