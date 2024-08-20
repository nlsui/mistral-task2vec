# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


#!/usr/bin/env python3.6
import pickle

from task2vec import Task2Vec
import yaml


def main():
    with open('conf/config.yaml', 'r') as file:
        config = yaml.safe_load(file)


if __name__ == "__main__":
    main()
