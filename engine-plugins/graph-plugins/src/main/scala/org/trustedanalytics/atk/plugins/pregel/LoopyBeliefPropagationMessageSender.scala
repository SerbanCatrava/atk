package org.trustedanalytics.atk.plugins.pregel

import org.apache.spark.graphx._
import org.trustedanalytics.atk.plugins.VectorMath
import org.trustedanalytics.atk.plugins.pregel.core.{ DefaultValues, VertexState }

object LoopyBeliefPropagationMessageSender {
  /**
   * Pregel required method to send messages across an edge.
   * @param edgeTriplet Contains state of source, destination and edge.
   * @return Iterator over messages to send.
   */
  def sendMessage(edgeTriplet: EdgeTriplet[VertexState, Double]): Iterator[(VertexId, Map[Long, Vector[Double]])] = {

    Iterator((edgeTriplet.dstId,
      calculateMessage(edgeTriplet.srcId, edgeTriplet.dstId, edgeTriplet.srcAttr, edgeTriplet.attr)))
  }

  /**
   * Calculates the message to be sent from one vertex to another.
   * @param sender ID of he vertex sending the message.
   * @param destination ID of the vertex to receive the message.
   * @param vertexState State of the sending vertex.
   * @param edgeWeight Weight of the edge joining the two vertices.
   * @return A map with one entry, sender -> messageToNeighbor
   */
  private def calculateMessage(sender: VertexId,
                               destination: VertexId,
                               vertexState: VertexState,
                               edgeWeight: Double): Map[VertexId, Vector[Double]] = {

    val prior = vertexState.prior
    val messages = vertexState.messages

    val nStates = prior.length
    val stateRange = (0 to nStates - 1).toVector

    val messagesNotFromDestination = messages - destination
    val messagesNotFromDestinationValues: List[Vector[Double]] =
      messagesNotFromDestination.map({ case (k, v) => v }).toList

    val reducedMessages = VectorMath.overflowProtectedProduct(prior :: messagesNotFromDestinationValues).get
    val statesUnPosteriors = stateRange.zip(reducedMessages)
    val unnormalizedMessage = stateRange.map(i => statesUnPosteriors.map({
      case (j, x: Double) =>
        x * Math.exp(edgePotential(i, j, edgeWeight))
    }).sum)

    val message = VectorMath.l1Normalize(unnormalizedMessage)

    Map(sender -> message)
  }

  /**
   * The edge potential function provides an estimate of how compatible the states are between two joined vertices.
   * This is the one inspired by the Boltzmann distribution.
   * @param state1 State of the first vertex.
   * @param state2 State of the second vertex.
   * @param weight Edge weight.
   * @return Compatibility estimate for the two states..
   */
  private def edgePotential(state1: Int, state2: Int, weight: Double) = {

    val compatibilityFactor =
      if (DefaultValues.powerDefault == 0d) {
        if (state1 == state2)
          0d
        else
          1d
      }
      else {
        val delta = Math.abs(state1 - state2)
        Math.pow(delta, DefaultValues.powerDefault)
      }

    -1.0d * compatibilityFactor * weight * DefaultValues.smoothingDefault
  }

}