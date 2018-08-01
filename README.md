
# AVNTests
AVN Correlator Unit and Acceptance Testing based Framework

## Installation

### Debian packages

Install dependencies:

    $ # sudo apt-get update && sudo apt-get install -yfm $(cat apt-build-requirements.txt)

### Python Core packages

Install dependencies to the system, by following their installation instructions:

* [_katcp-python_](https://github.com/ska-sa/katcp-python)
* [_nosekatreport_](https://github.com/ska-sa/nosekatreport/tree/karoocbf)

### Python testing dependencies packages

It is highly advisable to install these dependencies on a [_Python virtual environment_](https://virtualenv.pypa.io/), below is step-by-step instructions.
#### Setup and Use Virtualenv
```
# Install Python essentials and pip
$ curl -s https://bootstrap.pypa.io/get-pip.py | python
$ sudo pip install virtualenv virtualenvwrapper

# Install testing dependencies in the virtualenv
$ virtualenv venv
$ source ~/venv/bin/activate
$ cd AVNTests
$ pip install -r pip-requirements

```

### AVNTests installation (Optional)
Install AVN Tests 
```
$ . venv/bin/activate
$ python setup.py install
```

## Unit Testing

Running unit-testing.
```
$ . venv/bin/activate
# to follow...
```

## Contributors

 * Mpho Mphego
