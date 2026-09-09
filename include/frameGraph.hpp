#pragma once

namespace Brassica {
/*
	The basic notion is to have phases and graphs both be able to act like nodes, so that
	the graph can hold a group of them, and sort the phases in order, and nodes can specify
	that they come before or after different phases, and graphs can be treated as nodes so
	that complex nodes can themselves be comprised of nodes.
*/
	template <typename T> concept NodeLike = requires(){ true; };
	class Phase {};
	class Graph{};
	class Node{};
	class Queue{};
	class Resource{
		void GetReadBarrier() {};
		void GetReleaseBarrier(Queue destQueue) {};
		void GetAcquireBarrier(Queue sourceQueue) {};
	};
	class Recipe{};
	class PreviousFrame : public Node {};
	class NextFrame : public Node {};
	class FrameStart : public Phase {};
	class FrameEnd : public Phase {};
	class Operation {};
	template <typename T> class Create : public operation {};
	template <typename T> class Read : public operation {};
	template <typename T, typename X = T> class Modify : public operation {};
}; // namespace Brassica

