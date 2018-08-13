
# AVNTests
[African VLBI Network](https://www.ska.ac.za/science-engineering/avn/) (AVN) Correlator Unit and Acceptance Testing based Framework

## Installation

### Python Core packages

Install dependencies to the system, by following their installation instructions:

* [_katcp-python_](https://github.com/ska-sa/katcp-python)
* [_nosekatreport_](https://github.com/ska-sa/nosekatreport/tree/karoocbf)

### Python testing dependencies packages

It is highly recommended to install dependencies on a [_Python virtual environment_](https://virtualenv.pypa.io/).

#### Setup Virtualenv
Step-by-step instructions.
```
# Install Python essentials and pip
$ curl -s https://bootstrap.pypa.io/get-pip.py | python
$ pip install --user virtualenv # or sudo pip install virtualenv

# Install testing dependencies in the virtualenv
$ git clone https://github.com/ska-sa/AVNTests
$ cd AVNTests
$ make bootstrap
```

## Unit Testing

Running unit-testing.
```
$ make channelisation # This will run channelisation (Defined) test only
```

## Contributors

 * Mpho Mphego
