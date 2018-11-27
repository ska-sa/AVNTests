
# AVNTests

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/65abea0d64554c5ab27b6aad07496c2d)](https://app.codacy.com/app/mmphego/AVNTests?utm_source=github.com&utm_medium=referral&utm_content=ska-sa/AVNTests&utm_campaign=Badge_Grade_Settings)

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

## Usage

You will need to create a `$(pwd)/.env` file which contains:

```shell
touch .env 

echo "USERNAME=<Username on Host - usually: avn>" >> "$(pwd)/.env"
echo "PASSWORD=<Password>" >> "$(pwd)/.env"
echo "HOSTIP=<Signal Generator IP>" >> "$(pwd)/.env"
echo "KATCPIP=<katcp host IP>" >> "$(pwd)/.en"v
```

## Unit Testing

Running unit-testing.
```shell
make channelisation # This will run channelisation (Defined) test only
```

## Contributors

 * Mpho Mphego
 
 
## Feedback

Feel free to fork it or send me PR to improve it.

