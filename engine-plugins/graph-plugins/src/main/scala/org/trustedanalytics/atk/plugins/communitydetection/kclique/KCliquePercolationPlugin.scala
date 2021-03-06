/**
 *  Copyright (c) 2015 Intel Corporation 
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.trustedanalytics.atk.plugins.communitydetection.kclique

import org.trustedanalytics.atk.engine.plugin._
import org.trustedanalytics.atk.domain.frame.{ FrameReference, FrameEntity }
import org.trustedanalytics.atk.domain.CreateEntityArgs
import org.trustedanalytics.atk.domain.graph.GraphReference
import org.trustedanalytics.atk.engine.graph.SparkGraph
import org.apache.spark.frame.FrameRdd

/**
 * Represents the arguments for KClique Percolation algorithm
 *
 * @param graph Reference to the graph for which communities has to be determined.
 */
case class KCliqueArgs(graph: GraphReference,
                       @ArgDoc("""The sizes of the cliques used to form communities.
Larger values of clique size result in fewer, smaller communities that are more connected.
Must be at least 2.""") cliqueSize: Int,
                       @ArgDoc("""Name of the community property of vertex that will be updated/created in the graph.
This property will contain for each vertex the set of communities that contain that
vertex.""") communityPropertyLabel: String) {
  require(cliqueSize > 1, "Invalid clique size; must be at least 2")
}

case class KCliqueResult(frameDictionaryOutput: Map[String, FrameReference], time: Double)

/**
 * Json conversion for arguments and return value case classes
 */

object KCliquePercolationJsonFormat {
  import org.trustedanalytics.atk.domain.DomainJsonProtocol._
  implicit val kcliqueFormat = jsonFormat3(KCliqueArgs)
  implicit val kcliqueResultFormat = jsonFormat2(KCliqueResult)
}

import KCliquePercolationJsonFormat._
/**
 * KClique Percolation plugin class.
 */

@PluginDoc(oneLine = "Find groups of vertices with similar attributes.",
  extended = """**Community Detection Using the K-Clique Percolation Algorithm**

**Overview**

Modeling data as a graph captures relations, for example, friendship ties
between social network users or chemical interactions between proteins.
Analyzing the structure of the graph reveals collections (often termed
'communities') of vertices that are more likely to interact amongst each
other.
Examples could include a community of friends in a social network or a
collection of highly interacting proteins in a cellular process.

|PACKAGE| provides community detection using the k-Clique
percolation method first proposed by Palla et. al. [1]_ that has been widely
used in many contexts.

**K-Clique Percolation**

K-clique percolation is a method for detecting community structure in graphs.
Here we provide mathematical background on how communities are defined in the
context of the k-clique percolation algorithm.

A clique is a group of vertices in which every vertex is connected (via
undirected edge) with every other vertex in the clique.
This graphically looks like a triangle or a structure composed of triangles:

.. image:: /k-clique_201508281155.*

A clique is certainly a community in the sense that its vertices are all
connected, but, it is too restrictive for most purposes,
since it is natural some members of a community may not interact.

Mathematically, a k-clique has :math:`k` vertices, each with :math:`k - 1`
common edges, each of which connects to another vertex in the k-clique.
The k-clique percolation method forms communities by taking unions of k-cliques
that have :math:`k - 1` vertices in common.

**K-Clique Example**

In the graph below, the cliques are the sections defined by their triangular
appearance and the 3-clique communities are {1, 2, 3, 4} and {4, 5, 6, 7, 8}.
The vertices 9, 10, 11, 12 are not in 3-cliques, therefore they do not belong
to any community.
Vertex 4 belongs to two distinct (but overlapping) communities.

.. image:: /ds_mlal_a1.png

**Distributed Implementation of K-Clique Community Detection**

The implementation of k-clique community detection in |PACKAGE| is a fully
distributed implementation that follows the map-reduce
algorithm proposed in Varamesh et. al. [2]_ .

It has the following steps:

#.  All k-cliques are :term:`enumerated <enumerate>`.
#.  k-cliques are used to build a "clique graph" by declaring each k-clique to
    be a vertex in a new graph and placing edges between k-cliques that share
    k-1 vertices in the base graph.
#.  A :term:`connected component` analysis is performed on the clique graph.
    Connected components of the clique graph correspond to k-clique communities
    in the base graph.
#.  The connected components information for the clique graph is projected back
    down to the base graph, providing each vertex with the set of k-clique
    communities to which it belongs.

Notes
-----
Spawns a number of Spark jobs that cannot be calculated before execution
(it is bounded by the diameter of the clique graph derived from the input graph).
For this reason, the initial loading, clique enumeration and clique-graph
construction steps are tracked with a single progress bar (this is most of
the time), and then successive iterations of analysis of the clique graph
are tracked with many short-lived progress bars, and then finally the
result is written out.


.. rubric:: Footnotes

.. [1]
    G. Palla, I. Derenyi, I. Farkas, and T. Vicsek. Uncovering the overlapping
    community structure of complex networks in nature and society.
    Nature, 435:814, 2005 ( See http://hal.elte.hu/cfinder/wiki/papers/communitylettm.pdf )

.. [2]
    Varamesh, A.; Akbari, M.K.; Fereiduni, M.; Sharifian, S.; Bagheri, A.,
    "Distributed Clique Percolation based community detection on social
    networks using MapReduce,"
    Information and Knowledge Technology (IKT), 2013 5th Conference on, vol.,
    no., pp.478,483, 28-30 May 2013
""",
  returns = "Dictionary of vertex label and frame, Execution time."
)
class KCliquePercolationPlugin extends SparkCommandPlugin[KCliqueArgs, KCliqueResult] {

  /**
   * The name of the command, e.g. graphs/kclique_percolation
   */
  override def name: String = "graph:/kclique_percolation"

  override def apiMaturityTag = Some(ApiMaturityTag.Alpha)

  /**
   * The number of jobs varies with the number of supersteps required to find the connected components
   * of the derived clique-shadow graph.... we cannot properly anticipate this without doing a full analysis of
   * the graph.
   *
   * @param arguments command arguments: used if a command can produce variable number of jobs
   * @return number of jobs in this command
   */
  override def numberOfJobs(arguments: KCliqueArgs)(implicit invocation: Invocation): Int = {
    8 + 2 * arguments.cliqueSize
  }

  override def kryoRegistrator: Option[String] = None

  override def execute(arguments: KCliqueArgs)(implicit invocation: Invocation): KCliqueResult = {
    import org.trustedanalytics.atk.graphbuilder.rdd.GraphBuilderRddImplicits._

    val start = System.currentTimeMillis()

    // Get the graph
    val graph: SparkGraph = arguments.graph
    val (gbVertices, gbEdges) = graph.gbRdds
    val (outVertices, outEdges) = KCliquePercolationRunner.run(gbVertices, gbEdges, arguments.cliqueSize, arguments.communityPropertyLabel)

    val mergedVertexRdd = (outVertices ++ gbVertices).mergeDuplicates()

    // Get the execution time and print it
    val time = (System.currentTimeMillis() - start).toDouble / 1000.0

    val frameRddMap = FrameRdd.toFrameRddMap(mergedVertexRdd)

    val frameMap = frameRddMap.keys.map(label => {
      val result: FrameReference = engine.frames.tryNewFrame(CreateEntityArgs(description = Some("created by connected components operation"))) { newOutputFrame: FrameEntity =>
        val frameRdd = frameRddMap(label)
        newOutputFrame.save(frameRdd)
      }
      (label, result)
    }).toMap
    KCliqueResult(frameMap, time)
  }

}
