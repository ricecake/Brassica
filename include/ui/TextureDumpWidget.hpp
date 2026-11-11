#pragma once

#include <string>

#include "ui/IWidget.hpp"

namespace brassica::ui {

	// Debug panel for TextureDumpService -- one button per pre-registered world-state texture,
	// dumping it to debug_dumps/ as a PNG when clicked. Extending the list lives entirely in
	// Engine's own RegisterDumpable<K>() calls, not here -- this just draws whatever
	// TextureDumpService::Entries() currently has.
	class TextureDumpWidget: public IWidget {
	public:
		void Draw() override;

		[[nodiscard]] std::string_view GetTitle() const override { return "Texture Dump"; }

		[[nodiscard]] std::string_view GetCategory() const override { return "Debug"; }

	private:
		std::string m_lastResult;
	};

} // namespace brassica::ui
