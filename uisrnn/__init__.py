# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The module for Unbounded Interleaved-State Recurrent Neural Network.

An introduction is available at [README.md].

[README.md]: https://github.com/google/uis-rnn/blob/master/README.md
"""

from . import arguments
from . import evals
from . import loss_func
from . import uisrnn
from . import utils

#pylint: disable=C0103
parse_arguments = arguments.parse_arguments
compute_sequence_match_accuracy = evals.compute_sequence_match_accuracy
output_result = utils.output_result
UISRNN = uisrnn.UISRNN
