#include "passes/BallNode.hpp"

namespace brassica {

	BallPushConstants        BallNode::s_currentPush{};
	MeshTasksIndirectCommand BallNode::s_indirectCmd{};

} // namespace brassica
