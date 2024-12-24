import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
import json
from typing import List, Dict, Any, Optional, Set
import os
from collections import defaultdict
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

# Define Pydantic models for structured output parsing
class RelevantFact(BaseModel):
    entity: str = Field(description="Entity name")
    relation: str = Field(description="Relationship type")
    value: str = Field(description="Value or target entity")
    sub_objective_index: int = Field(description="Index of related sub-objective")
    relevance: str = Field(description="Relevance level: high/medium/low")
    reasoning: str = Field(description="Why this fact is relevant")

class NextEntity(BaseModel):
    entity: str = Field(description="Entity to explore next")
    reason: str = Field(description="Reason for exploring this entity")
    priority: str = Field(description="Priority level: high/medium/low")

class ExplorationResult(BaseModel):
    relevant_facts: List[RelevantFact]
    next_entities: List[NextEntity]
    reasoning: str

# Knowledge Graph Structure
KNOWLEDGE_GRAPH = {
    "Claude_Debussy": {
        "music.artist.genre": ["Ballet", "Impressionism"],
        "music.artist.composed": ["Afternoon_of_a_Faun", "Jeux"],
        "music.style.known_for": ["Ballet_compositions", "Orchestral_works"]
    },
    "Suzanne_Farrell_Elusive_Muse": {
        "film.film.genre": ["Documentary", "Ballet"],
        "film.film.features": ["Ballet_Music", "Classical_Dance"],
        "film.film.soundtrack": ["Ballet_Music_By_Debussy", "Other_Classical_Works"]
    },
    "Ballet": {
        "music.genre.associated_with": ["Claude_Debussy", "Igor_Stravinsky"],
        "art.form.characteristics": ["Dance", "Classical_Music", "Performance"],
        "music.style.composers": ["Debussy", "Tchaikovsky", "Stravinsky"]
    },
    "The_Naked_and_the_Dead": {
        "film.film.setting": ["Panama"],
        "film.film.time_period": ["World_War_II"],
        "film.film.location": ["Pacific_Theater"]
    },
    "Panama": {
        "location.country.leader": ["President_of_Panama"],
        "location.country.government": ["Presidential_Republic"],
        "location.country.capital": ["Panama_City"]
    },
    "President_of_Panama": {
        "government.role.holder": ["Juan_Carlos_Varela"],
        "government.role.jurisdiction": ["Panama"],
        "government.role.period": ["2014_2019"]
    }
}

KNOWLEDGE_GRAPH.update({
    "Suzanne_Farrell_Elusive_Muse": {
        "film.film.genre": ["Documentary", "Ballet"],
        "film.film.featured_music": ["Ballet_Music"],
        "film.film.year": "1996",
        "film.film.subject": ["New_York_City_Ballet", "Ballet_History"]
    },
    "World_War_II_Panama": {
        "location.historical.period": "1939-1945",
        "location.historical.control": ["US_Military_Forces"],
        "location.historical.context": ["Pacific_Theater", "Canal_Zone"]
    }
})

# Add historical context for temporal reasoning
HISTORICAL_CONTEXT = {
    "World_War_II_Panama": {
        "location.historical.period": "1939-1945",
        "location.historical.control": ["US_Military_Forces"],
        "location.historical.context": ["Pacific_Theater", "Canal_Zone"]
    }
}

@dataclass
class ExplorationState:
    """
    Section 3.2: Path Exploration Implementation

    Tracks the current state of knowledge graph exploration:
    - Current entity being explored
    - Path taken through the graph
    - Facts discovered during exploration
    """
    current_entity: str
    path: List[str]
    found_facts: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EntityExtraction:
    main_entities: List[str]
    context_entities: List[str]

@dataclass
class TemporalContext:
    """
    Handles temporal reasoning for historical facts:
    - Ensures temporal consistency in reasoning
    - Validates facts against time periods
    - Helps avoid anachronistic conclusions
    """
    time_period: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@dataclass
class SubObjective:
    index: int
    description: str
    status: str = ""
    completed: bool = False
    found_facts: List[Dict[str, Any]] = field(default_factory=list)

    def add_fact(self, entity: str, relation: str, value: str):
        self.found_facts.append({
            "entity": entity,
            "relation": relation,
            "value": value
        })
        self.update_status()

    def update_status(self):
        if self.found_facts:
            self.status = f"Found {len(self.found_facts)} relevant facts"
            self.completed = True
        else:
            self.status = "Pending exploration"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "status": self.status,
            "completed": self.completed,
            "found_facts": self.found_facts
        }


class PoG:
    def __init__(self):
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY environment variable not set")
            
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )
        self.output_parser = PydanticOutputParser(pydantic_object=ExplorationResult)
        self.knowledge_graph = KNOWLEDGE_GRAPH

    def explore_entity(self, entity: str, sub_objectives: List[Dict]) -> ExplorationResult:
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following entity and determine relevance to objectives.
        Entity: {entity}
        Available Information: {entity_info}
        Connected Entities: {connected_entities}
        Sub-objectives: {sub_objectives}

        {format_instructions}
        """)

        # Get format instructions from the parser
        format_instructions = self.output_parser.get_format_instructions()

        try:
            result = self.llm.invoke(
                prompt.format(
                    entity=entity,
                    entity_info=self.get_entity_info(entity),
                    connected_entities=self.get_connected_entities(entity),
                    sub_objectives=sub_objectives,
                    format_instructions=format_instructions
                )
            )
            return self.output_parser.parse(result.content)
        except Exception as e:
            st.error(f"Error exploring entity {entity}: {str(e)}")
            return ExplorationResult(relevant_facts=[], next_entities=[], reasoning="Error in exploration")

    def get_connected_entities(self, entity: str) -> List[str]:
        connected = set()
        entity_info = self.get_entity_info(entity)
        for values in entity_info.values():
            if isinstance(values, list):
                for value in values:
                    if self.normalize_entity(value) in self.knowledge_graph:
                        connected.add(value)
        return list(connected)
    

    def reflect_and_plan(self, state: ExplorationState, sub_objectives: List[SubObjective]) -> Dict[str, Any]:
        """
        Section 3.4: Reflection Mechanism

        Implements:
        1. Progress evaluation
        2. Self-correction triggering
        3. Exploration strategy adjustment
        """
        prompt = f"""Evaluate progress and determine if correction is needed.
        Return format:
        {{
            "progress_assessment": {{
                "completed_objectives": [0, 1],
                "pending_objectives": [2, 3],
                "sufficient_information": false
            }},
            "correction_needed": false,
            "correction_strategy": null,
            "next_steps": ["specific action"],
            "reasoning": "detailed explanation"
        }}

        Current Path: {state.path}
        Found Facts: {json.dumps(state.found_facts, indent=2)}
        Sub-objectives Progress: {[obj.to_dict() for obj in sub_objectives]}"""

        return self._call_llm(prompt, "Expert in reasoning evaluation")

    def answer_question(self, question: str) -> str:
        """
        Main PoG pipeline implementing the paper's workflow:
        1. Task decomposition (Section 3.1)
        2. Adaptive exploration (Section 3.2)
        3. Memory updating (Section 3.3)
        4. Reflection and correction (Section 3.4)
        """
        print(f"\nQuestion: {question}")
        print("=" * 80)

        # Step 1: Extract entities
        entities = self.extract_entities(question)
        print("\nIdentified Entities:")
        print(f"Main: {entities.main_entities}")
        print(f"Context: {entities.context_entities}")

        # Step 2: Task Decomposition
        print("\n1. Decomposing into Sub-objectives:")
        self.sub_objectives = self.decompose_task(question)
        for i, obj in enumerate(self.sub_objectives):
            print(f"  {i+1}. {obj.description}")

        # Initialize exploration with main entities
        explored_entities = set()
        to_explore = entities.main_entities.copy()
        facts_found = []

        max_depth = 4
        depth = 0

        while to_explore and depth < max_depth:
            current_entity = to_explore.pop(0)
            print(f"\n{depth + 2}. Exploring {current_entity}...")

            if self.normalize_entity(current_entity) not in self.knowledge_graph:
                print(f"No information found for {current_entity}")
                continue

            # Explore current entity
            exploration = self.explore_entity(current_entity, self.sub_objectives)
            explored_entities.add(self.normalize_entity(current_entity))

            print("Found relevant facts:")
            #for fact in exploration.get("relevant_facts", []):
            for fact in exploration.relevant_facts:
                # print(f"  - {fact['relation']}: {fact['value']}")
                # print(f"    Relevance: {fact['relevance']}")
                # print(f"    Reasoning: {fact['reasoning']}")
                # facts_found.append(fact)
                print(f"  - {fact.relation}: {fact.value}")
                print(f"    Relevance: {fact.relevance}")
                print(f"    Reasoning: {fact.reasoning}")
                facts_found.append({
                    "entity": fact.entity,
                    "relation": fact.relation,
                    "value": fact.value,
                    "sub_objective_index": fact.sub_objective_index
                })
                # Update sub-objective completion
                # obj_idx = fact["sub_objective_index"]
                obj_idx = fact.sub_objective_index
                if obj_idx < len(self.sub_objectives):
                    self.sub_objectives[obj_idx].add_fact(
                        # fact["entity"],
                        # fact["relation"],
                        # fact["value"]
                        fact.entity,
                        fact.relation,
                        fact.value
                    )

            print(f"\nReasoning: {exploration.reasoning}")

            # Add new entities to explore
            #for next_entity in exploration.get("next_entities", []):
            for next_entity in exploration.next_entities:
                entity_name = self.normalize_entity(next_entity.entity)
                if (entity_name in self.knowledge_graph and
                    entity_name not in explored_entities and
                    entity_name not in [self.normalize_entity(e) for e in to_explore]):
                    print(f"Adding {next_entity.entity} to explore (Priority: {next_entity.priority})")
                    if next_entity.priority == 'high':
                        to_explore.insert(0, next_entity.entity)
                    else:
                        to_explore.append(next_entity.entity)

            depth += 1

        # Construct answer from collected facts
        if facts_found:
            answer = self.construct_answer(question, self.sub_objectives, facts_found)
            return answer
        else:
            return "Unable to find relevant information to answer the question."

    def extract_entities(self, question: str) -> EntityExtraction:
        """Extract main and context entities from the question"""
        prompt = f"""Identify the main entities and context entities from this question. Return format:
        {{
            "main_entities": ["primary entity 1", "primary entity 2"],
            "context_entities": ["context entity 1", "context entity 2"]
        }}

        Question: {question}"""

        response = self._call_llm(prompt, "You are an expert in entity extraction from text.")

        # Clean and normalize entity names
        main_entities = [self.normalize_entity(e) for e in response["main_entities"]]
        context_entities = [self.normalize_entity(e) for e in response["context_entities"]]

        return EntityExtraction(main_entities, context_entities)

    def normalize_entity(self, entity: str) -> str:
        if not isinstance(entity, str):
            return ""
        return entity.replace(" ", "_").replace("'", "").replace('"', '')

    def get_entity_info(self, entity: str) -> Dict[str, List[str]]:
        normalized_entity = self.normalize_entity(entity)
        return self.knowledge_graph.get(normalized_entity, {})
    
    def decompose_task(self, question: str) -> List[SubObjective]:
        """Decompose question into focused sub-objectives"""
        prompt = f"""Break down this question into specific, focused sub-objectives. Return format:
        {{
            "sub_objectives": [
                {{
                    "description": "specific task description",
                    "required_info": ["what information is needed"]
                }}
            ]
        }}

        Question: {question}"""

        response = self._call_llm(prompt, "You are an expert in breaking down complex questions.")

        # Create SubObjective instances with index
        self.sub_objectives = [
            SubObjective(index=i, description=obj["description"])
            for i, obj in enumerate(response["sub_objectives"])
        ]
        return self.sub_objectives

    def get_next_entity(self, current_entity: str, exploration_result: Dict[str, Any],
                        explored_entities: Set[str]) -> Optional[str]:
        """Determine next entity to explore based on knowledge graph connections"""
        if not exploration_result.get("next_entities"):
            # Look for unexplored connected entities
            connected_entities = set(exploration_result["connected_entities"])
            unexplored = connected_entities - explored_entities
            if unexplored:
                return unexplored.pop()

            # If no unexplored connected entities, look at parent entities
            for entity, info in self.knowledge_graph.items():
                for values in info.values():
                    if current_entity in values and entity not in explored_entities:
                        return entity
        else:
            # Use suggested next entities
            for next_entity in exploration_result["next_entities"]:
                entity_name = next_entity["entity"]
                if entity_name not in explored_entities and entity_name in self.knowledge_graph:
                    return entity_name

        return None

    def check_temporal_relevance(self, fact: Dict[str, Any], context: TemporalContext) -> bool:
        """Check if fact is relevant to the temporal context"""
        if fact.get("period"):
            return self.dates_overlap(fact["period"], context)
        return True

#     def construct_answer(self, question: str, sub_objectives: List[SubObjective],
#                         facts: List[Dict[str, Any]]) -> str:
#         """Construct final answer from collected facts with better organization"""
#         # Group facts by sub-objective
#         facts_by_objective = defaultdict(list)
#         for fact in facts:
#             obj_idx = fact.get("sub_objective_index", 0)
#             facts_by_objective[obj_idx].append(fact)

#         # Prepare facts summary for each sub-objective
#         objectives_summary = []
#         for obj in sub_objectives:
#             relevant_facts = facts_by_objective[obj.index]
#             objectives_summary.append({
#                 "objective": obj.description,
#                 "facts": relevant_facts,
#                 "completed": bool(relevant_facts)
#             })

#         prompt = f"""Based on the collected facts, construct a clear answer to the question.

#         Question: {question}

#         Evidence gathered:
#         {json.dumps(objectives_summary, indent=2)}

#         Return format:
#         {{
#             "answer": "clear and concise answer based on the evidence",
#             "reasoning": "explanation of how the facts support this answer",
#             "confidence": "high/medium/low based on evidence completeness"
#         }}"""

#         response = self._call_llm(prompt, "You are an expert in synthesizing information to answer questions.")

#         # Format the answer with confidence level
#         confidence = response.get("confidence", "medium")
#         answer = response["answer"]
#         reasoning = response["reasoning"]

#         return f"""Answer ({confidence} confidence): {answer}

# Reasoning: {reasoning}

# Based on evidence from:
# {self._format_evidence_summary(objectives_summary)}"""

    def construct_answer(self, question: str, sub_objectives: List[SubObjective], facts: List[Dict[str, Any]]) -> str:
        # Group facts by sub-objective
        facts_by_objective = defaultdict(list)
        for fact in facts:
            obj_idx = fact.get("sub_objective_index", 0)
            facts_by_objective[obj_idx].append(fact)

        objectives_summary = []
        for obj in sub_objectives:
            relevant_facts = facts_by_objective[obj.index]
            objectives_summary.append({
                "objective": obj.description,
                "facts": relevant_facts,
                "completed": bool(relevant_facts)
            })

        # prompt = f"""Based on the collected facts, construct a clear answer to the question in a json format
        # Question: {question}
        # Evidence gathered: {json.dumps(objectives_summary, indent=2)}"""
        prompt = f"""Based on the collected facts, return only json output consisting of only key i.e 'answer'.
        For your reference find below:
        Question: {question}
        Evidence gathered: {json.dumps(objectives_summary, indent=2)}
        """
        try:
            response = self._call_llm(prompt, "You are an expert in synthesizing information to answer questions.")
            if "error" in response:
                return "Unable to construct answer due to parsing error"
                
            return f"""Answer: {response.get('answer', 'No answer available')}
            
                    Evidence from:
                    {self._format_evidence_summary(objectives_summary)}"""
        except Exception as e:
            return f"Error constructing answer: {str(e)}"
    
    def _format_evidence_summary(self, objectives_summary: List[Dict[str, Any]]) -> str:
        """Format evidence summary for output"""
        lines = []
        for obj in objectives_summary:
            status = "✓" if obj["completed"] else "×"
            facts_count = len(obj["facts"])
            lines.append(f"{status} {obj['objective']} ({facts_count} facts)")
        return "\n".join(lines)

    # def _call_llm(self, prompt: str, system_message: str) -> Dict[str, Any]:
    #     """Call LLM with proper error handling"""
    #     messages = [
    #         {"role": "system", "content": f"You are an AI assistant that always responds in JSON format. {system_message}"},
    #         {"role": "user", "content": f"{prompt}\nRespond in JSON format."}
    #     ]

    #     try:
    #         response = self.llm.invoke(prompt)
    #         return json.loads(response.content)
    #     except Exception as e:
    #         print(f"Error in LLM call: {e}")
    #         print(f"Prompt that caused error: {prompt}")
    #         raise
    def _call_llm(self, prompt: str, system_message: str) -> Dict[str, Any]:
        try:
            response = self.llm.invoke(f"System: {system_message}\n\nHuman: {prompt}")
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                print(f"Raw response: {response.content}")  # Debug line
                return {"error": "Failed to parse JSON response"}
        except Exception as e:
            print(f"Error in LLM call: {e}")
            print(f"Prompt that caused error: {prompt}")
            raise
def main():
    st.set_page_config(page_title="Plan-on-Graph Explorer", layout="wide")
    
    with st.sidebar:
        st.header("Configuration")
        #api_key = st.text_input("Groq API Key", type="password")
        api_key = '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'
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

    question = st.text_input("Enter your question:", key="question_input")

    if st.button("Explore", disabled=not api_key):
        if not api_key:
            st.error("Please enter your Groq API Key")
            return

        try:
            with st.spinner("Processing..."):
                pog = PoG()
                result = pog.answer_question(question)
                st.write(result)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()