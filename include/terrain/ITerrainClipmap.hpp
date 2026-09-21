#pragma once

#include "IManager.hpp"

namespace brassica {

	class ITerrainClipmap: public IManager {
	public:
		~ITerrainClipmap() override = default;

		virtual void Regenerate() = 0;
	};

} // namespace brassica
