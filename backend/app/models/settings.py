from pydantic import BaseModel, Field
from typing import Optional, Literal


Theme = Literal['light', 'dark']
LayoutName = Literal['cose', 'grid', 'circle']


class AppSettings(BaseModel):
    id: str = "app"
    theme: Theme = Field(default='light')
    show_edge_labels: bool = Field(default=True)
    default_relation_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    layout_name: LayoutName = Field(default='cose')
    animate_layout: bool = Field(default=False)


class AppSettingsUpdate(BaseModel):
    theme: Optional[Theme] = None
    show_edge_labels: Optional[bool] = None
    default_relation_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    layout_name: Optional[LayoutName] = None
    animate_layout: Optional[bool] = None


