"""
Cross-platform symbols and emoji with ASCII fallbacks
"""

import sys


def _can_encode(text: str) -> bool:
    """Check if text can be encoded with current stdout encoding"""
    try:
        text.encode(sys.stdout.encoding)
        return True
    except (UnicodeEncodeError, AttributeError):
        return False


# Define symbols with fallbacks
class Symbols:
    """Cross-platform symbols"""
    
    # Success/failure
    CHECK = "✓" if _can_encode("✓") else "[OK]"
    CROSS = "✗" if _can_encode("✗") else "[X]"
    SUCCESS = "✅" if _can_encode("✅") else "[SUCCESS]"
    FAILURE = "❌" if _can_encode("❌") else "[FAILED]"
    WARNING = "⚠️" if _can_encode("⚠️") else "[WARNING]"
    
    # Progress
    HOURGLASS = "⏳" if _can_encode("⏳") else "[...]"
    TIMER = "⏱️" if _can_encode("⏱️") else "[TIME]"
    
    # Arrows
    ARROW_RIGHT = "→" if _can_encode("→") else "->"
    ARROW_LEFT = "←" if _can_encode("←") else "<-"
    ARROW_UP = "↑" if _can_encode("↑") else "^"
    ARROW_DOWN = "↓" if _can_encode("↓") else "v"
    
    # Files/folders
    FOLDER = "📂" if _can_encode("📂") else "[DIR]"
    FILE = "📄" if _can_encode("📄") else "[FILE]"
    
    # UI elements
    BULLET = "•" if _can_encode("•") else "*"
    INFO = "ℹ️" if _can_encode("ℹ️") else "[i]"
    
    # Generation
    HAMMER = "🔨" if _can_encode("🔨") else "[BUILD]"
    SPARKLES = "✨" if _can_encode("✨") else "[*]"
    ROBOT = "🤖" if _can_encode("🤖") else "[AI]"
    CHART = "📊" if _can_encode("📊") else "[DATA]"
    PENCIL = "📝" if _can_encode("📝") else "[WRITE]"
    SHIELD = "🛡️" if _can_encode("🛡️") else "[DEF]"


# Convenience exports
CHECK = Symbols.CHECK
CROSS = Symbols.CROSS
SUCCESS = Symbols.SUCCESS
FAILURE = Symbols.FAILURE
WARNING = Symbols.WARNING
HOURGLASS = Symbols.HOURGLASS
TIMER = Symbols.TIMER
ARROW_RIGHT = Symbols.ARROW_RIGHT
BULLET = Symbols.BULLET
HAMMER = Symbols.HAMMER
SPARKLES = Symbols.SPARKLES
ROBOT = Symbols.ROBOT
CHART = Symbols.CHART
PENCIL = Symbols.PENCIL
FOLDER = Symbols.FOLDER