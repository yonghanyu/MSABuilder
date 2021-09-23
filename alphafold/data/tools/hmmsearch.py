# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Python wrapper for hmmsearch - search profile against a sequence db."""

import os
import subprocess
from typing import Optional, Sequence

from absl import logging
from alphafold.data.tools import utils
# Internal import (7716).


class Hmmsearch(object):
  """Python wrapper of the hmmsearch binary."""

  def __init__(self,
               *,
               binary_path: str,
               database_path: str,
               param):
    """Initializes the Python hmmsearch wrapper.

    Args:
      binary_path: The path to the hmmsearch executable.
      database_path: The path to the hmmsearch database (FASTA format).
      flags: List of flags to be used by hmmsearch.

    Raises:
      RuntimeError: If hmmsearch binary not found within the path.
    """
    self.binary_path = binary_path
    self.database_path = database_path
    self.flags = param.flags
    self.n_cpu = param.n_cpu
    self.e_value = param.e_value
    self.z_value = param.z_value
    self.filter_f1 = param.filter_f1
    self.filter_f2 = param.filter_f2
    self.filter_f3 = param.filter_f3
    self.incdom_e = param.incdom_e
    self.dom_e = param.dom_e


    if not os.path.exists(self.database_path):
      logging.error('Could not find hmmsearch database %s', database_path)
      raise ValueError(f'Could not find hmmsearch database {database_path}')

  def query(self, hmm: str) -> str:
    """Queries the database using hmmsearch using a given hmm."""
    with utils.tmpdir_manager(base_dir='/tmp') as query_tmp_dir:
      hmm_input_path = os.path.join(query_tmp_dir, 'query.hmm')
      a3m_out_path = os.path.join(query_tmp_dir, 'output.a3m')
      with open(hmm_input_path, 'w') as f:
        f.write(hmm)

      cmd = [
          self.binary_path,
          '--noali',  # Don't include the alignment in stdout.
          '--F1', str(self.filter_f1),
          '--F2', str(self.filter_f2),
          '--F3', str(self.filter_f3),
          '--incE', str(self.e_value),
          # Report only sequences with E-values <= x in per-sequence output.
          '-E', str(self.e_value),
          '--cpu', str(self.n_cpu)
      ]
      # If adding flags, we have to do so before the output and input:
      if self.flags:
        cmd.extend(self.flags)
      if self.z_value:
        cmd.extend(['-Z', str(self.z_value)])

      if self.dom_e is not None:
        cmd.extend(['--domE', str(self.dom_e)])

      if self.incdom_e is not None:
        cmd.extend(['--incdomE', str(self.incdom_e)])

      cmd.extend([
          '-A', a3m_out_path,
          hmm_input_path,
          self.database_path,
      ])

      logging.info('Launching sub-process %s', cmd)
      process = subprocess.Popen(
          cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      with utils.timing(
          f'hmmsearch ({os.path.basename(self.database_path)}) query'):
        stdout, stderr = process.communicate()
        retcode = process.wait()

      if retcode:
        raise RuntimeError(
            'hmmsearch failed:\nstdout:\n%s\n\nstderr:\n%s\n' % (
                stdout.decode('utf-8'), stderr.decode('utf-8')))

      with open(a3m_out_path) as f:
        a3m_out = f.read()

    return a3m_out
