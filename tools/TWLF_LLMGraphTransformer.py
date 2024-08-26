from typing import Any, Dict, List, Optional, Tuple, Type, Union
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import default_prompt, _Graph, optional_enum_field, system_prompt
from pydantic.v1 import BaseModel, Field, create_model

from langchain_core.prompts import ChatPromptTemplate

class TWLF_LLMGraphTransformer(LLMGraphTransformer):
    '''
    1. 由於原生 LLMGraphTransformer node_properties 並不支援參數為必填(預設都是讓LLM選填)
        因此需要透過自定義來處理為每個 node_propertie 都為必填
    2. System Prompt 提示使用繁體中文回應問題
    
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        node_properties = kwargs.get("node_properties", [])
        allowed_nodes = kwargs.get("allowed_nodes", [])
        allowed_relationships = kwargs.get("allowed_relationships", [])
        llm = kwargs.get("llm", None)
        relationship_properties = kwargs.get("relationship_properties", [])
        prompt = kwargs.get("prompt", None)
        try:
            llm_type = llm._llm_type  # type: ignore
        except AttributeError:
            llm_type = None
        schema = my_create_simple_model(
            allowed_nodes,
            allowed_relationships,
            node_properties,
            llm_type,
            relationship_properties,
        )
    
        structured_llm = llm.with_structured_output(schema, include_raw=True)
        prompt = prompt or _get_prompt() # 新增使用繁體中文回應問題
        self.chain = prompt | structured_llm

def _get_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "system",
                "請確保使用繁體中文回答問題"
            ),
            (
                "human",
                (
                    "Tip: Make sure to answer in the correct format and do "
                    "not include any explanations. "
                    "Use the given format to extract information from the "
                    "following input: {input}"
                ),
            ),
        ]
    )
    return prompt

def my_create_simple_model(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[List[str]] = None,
    node_properties: Union[bool, List[str]] = False,
    llm_type: Optional[str] = None,
    relationship_properties: Union[bool, List[str]] = False,
) -> Type[_Graph]:
    """
    參考 langchain_experimental.graph_transformers.llm 的 create_simple_model, 基本只修改以下部分
    
    node_fields["properties"] = (
        Optional[List[Property]],
        Field(..., description="List of node properties") # 這裡從 None -> ...(變成必填) -> 只修改這行!
    )
    
    以下為原生註解:
        
    Create a simple graph model with optional constraints on node
    and relationship types.

    Args:
        node_labels (Optional[List[str]]): Specifies the allowed node types.
            Defaults to None, allowing all node types.
        rel_types (Optional[List[str]]): Specifies the allowed relationship types.
            Defaults to None, allowing all relationship types.
        node_properties (Union[bool, List[str]]): Specifies if node properties should
            be included. If a list is provided, only properties with keys in the list
            will be included. If True, all properties are included. Defaults to False.
        relationship_properties (Union[bool, List[str]]): Specifies if relationship
            properties should be included. If a list is provided, only properties with
            keys in the list will be included. If True, all properties are included.
            Defaults to False.
        llm_type (Optional[str]): The type of the language model. Defaults to None.
            Only openai supports enum param: openai-chat.

    Returns:
        Type[_Graph]: A graph model with the specified constraints.

    Raises:
        ValueError: If 'id' is included in the node or relationship properties list.
    """

    node_fields: Dict[str, Tuple[Any, Any]] = {
        "id": (
            str,
            Field(..., description="Name or human-readable unique identifier.")
        ),
        "type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
    }

    if node_properties:
        if isinstance(node_properties, list) and "id" in node_properties:
            raise ValueError("The node property 'id' is reserved and cannot be used.")
        # Map True to empty array
        node_properties_mapped: List[str] = (
            [] if node_properties is True else node_properties
        )

        class Property(BaseModel):
            """A single property consisting of key and value"""
            key: str = optional_enum_field(
                node_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="value")

        node_fields["properties"] = (
            Optional[List[Property]],
            Field(..., description="List of node properties") # 這裡從 None -> ...(變成必填)
        )
    # global SIMPLE_NODE
    SimpleNode = create_model("SimpleNode", **node_fields)  # type: ignore

    relationship_fields: Dict[str, Tuple[Any, Any]] = {
        "source_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of source node",
            )
        ),
        "source_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the source node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "target_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of target node",
            )
        ),
        "target_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the target node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "type": (
            str,
            optional_enum_field(
                rel_types,
                description="The type of the relationship.",
                input_type="relationship",
                llm_type=llm_type,
            ),
        ),
    }
    if relationship_properties:
        if (
            isinstance(relationship_properties, list)
            and "id" in relationship_properties
        ):
            raise ValueError(
                "The relationship property 'id' is reserved and cannot be used."
            )
        # Map True to empty array
        relationship_properties_mapped: List[str] = (
            [] if relationship_properties is True else relationship_properties
        )

        class RelationshipProperty(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                relationship_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="value")

        relationship_fields["properties"] = (
            Optional[List[RelationshipProperty]],
            Field(None, description="List of relationship properties")
        )
    SimpleRelationship = create_model("SimpleRelationship", **relationship_fields)  # type: ignore

    class DynamicGraph(_Graph):
        class Config:
            arbitrary_types_allowed = True
        """Represents a graph document consisting of nodes and relationships."""
        nodes: Optional[List[SimpleNode]] = Field(description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(  # type: ignore
            description="List of relationships"
        )

    return DynamicGraph