# Improved Side-Channels using Soft Decoding
This is some code for simulating the performance of using soft decoding techniques for improving side channel attacks (SCA) on the Fujisaka Okomoto transform (FO) on Post Quantum Cryptography primitives (PQC). This code accompanies the paper "SCA-LDPC: A Code-Based Framework for Key-Recovery Side-Channel Attacks on Post-quantum Encryption Schemes" (https://link.springer.com/chapter/10.1007/978-981-99-8730-6_7)

## Running python scripts
 Before running the script `main.py` please ensure that you are running in a virtual environment by executing

    cd simulate-with-python
    source setup-environment.sh

Afterwards the scripts in the `simulate-with-python` can be executed while being certain you are using the same dependencies as specified in `requirements.txt`

## Running rust program

This is just for experimentation, there is nothing concrete here yet. There might never be. It might never be necessary to use rust for these simulations.

<!--
## Description

*Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.*

## Badges
*On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.*

## Visuals
*Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.*

## Installation
*Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.*

## Usage
*Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.*

## Support
*Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.*

## Roadmap
*If you have ideas for releases in the future, it is a good idea to list them in the README.*

## Contributing
*State if you are open to contributions and what your requirements are for accepting them.*

*For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.*

*You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.*

## Authors and acknowledgment
*Show your appreciation to those who have contributed to the project.*

## License
*For open source projects, say how it is licensed.*
-->