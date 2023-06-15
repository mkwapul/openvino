# Command Line Tool to Run Ngraph Transformations on IR Graphs

## Introduction

During Intel® GNA graph compiler development, there may be times when new operators are not supported or a particular operator's parameters are not directly supported by Intel® GNA hardware.  Often, in these cases, it is possible to transform the IR graph to expand missing operators into Intel® GNA compatible subgraphs and factor incompatible operators into a subgraph of compatible operators.

This tool facilitates such IR transformations from the command line.  By specifying the name of the IR source graph and the ordered list of transformations to carry out, a new graph is produced whose name is the name of the source graph with "_factorized" appended to it.

Adding new ngraph transformations is easy.  So the hope is that this tool will enable quick fixes for customers while the Intel® GNA Plugin slowly evolves.

## How to Build

Create a new folder under samples/cpp called "run_ngraph".  Please copy CMakeLists.txt, \*.cpp, and \*.hpp into that folder before running cmake to create the solution (OpenVINO.sln).  Then run_ngraph may be built like any other sample.

## How to Run

For usage, run run_ngraph.exe without any parameters and a usage message will be generated listing all the available transformations and showing how to run them.

## Questions?

Contact michael.deisher@intel.com
