/*
// Copyright (c) 2015 Intel Corporation 
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

package org.trustedanalytics.atk.domain.frame

import org.trustedanalytics.atk.engine.plugin.{ ArgDoc, Invocation }

/**
 * Command to flatten the specified columns of the data frame and store the result to a
 * new data frame.
 */
case class FlattenColumnArgs(frame: FrameReference,
                             @ArgDoc("""The columns to be flattened.""") columns: List[String],
                             @ArgDoc("""The list of delimiter strings for each column.
Default is comma (,).""") delimiters: Option[List[String]] = None) {
  require(frame != null, "frame is required")
  require(columns != null && columns.nonEmpty, "column list is required")
  if (delimiters != null)
    require((delimiters.size == columns.size) || (delimiters.size == 1) || (delimiters.size == 0),
      "number of delimiters provided must match the number of columns")
}
