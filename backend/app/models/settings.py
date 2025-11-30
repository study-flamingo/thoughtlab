from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Literal as TypingLiteral


Theme = Literal['light', 'dark']
LayoutName = Literal['cose', 'grid', 'circle']


class RelationStyle(BaseModel):
    line_color: str = Field(default="#6B7280")  # gray-500
    target_arrow_color: Optional[str] = None
    width: int = Field(default=2, ge=1, le=12)
    line_style: Optional[TypingLiteral['solid', 'dashed', 'dotted']] = None
    target_arrow_shape: Optional[TypingLiteral['triangle', 'tee', 'none']] = None


class AppSettings(BaseModel):
    id: str = "app"
    theme: Theme = Field(default='light')
    show_edge_labels: bool = Field(default=True)
    default_relation_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    layout_name: LayoutName = Field(default='cose')
    animate_layout: bool = Field(default=False)

    # Visualization customization
    node_colors: Dict[str, str] = Field(
        default_factory=lambda: {
            "Observation": "#60A5FA",  # blue-400
            "Hypothesis": "#34D399",   # emerald-400
            "Source": "#FBBF24",       # amber-400
            "Concept": "#A78BFA",      # violet-400
            "Entity": "#F87171",       # red-400
        }
    )
    relation_styles: Dict[str, RelationStyle] = Field(
        default_factory=lambda: {
            "SUPPORTS": RelationStyle(line_color="#10B981", target_arrow_color="#10B981", width=3, target_arrow_shape="triangle"),
            "CONTRADICTS": RelationStyle(line_color="#EF4444", target_arrow_color="#EF4444", width=3, line_style="dashed", target_arrow_shape="tee"),
            "RELATES_TO": RelationStyle(line_color="#6B7280", target_arrow_color="#6B7280", width=2, target_arrow_shape="triangle"),
            "OBSERVED_IN": RelationStyle(line_color="#3B82F6", target_arrow_color="#3B82F6", width=2, target_arrow_shape="triangle"),
            "DISCUSSES": RelationStyle(line_color="#8B5CF6", target_arrow_color="#8B5CF6", width=2, target_arrow_shape="triangle"),
        }
    )


class AppSettingsUpdate(BaseModel):
    theme: Optional[Theme] = None
    show_edge_labels: Optional[bool] = None
    default_relation_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    layout_name: Optional[LayoutName] = None
    animate_layout: Optional[bool] = None
    node_colors: Optional[Dict[str, str]] = None
    relation_styles: Optional[Dict[str, RelationStyle]] = None


