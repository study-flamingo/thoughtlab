# Research Connection Graph - Project Overview

## Vision

A web-based research application that helps users discover meaningful connections between sources, observations, hypotheses, and concepts. The system uses graph database technology combined with LLM-powered analysis to automatically surface relationships that might otherwise go unnoticed.

## Core Problem

Researchers and knowledge workers frequently encounter situations where:
- An observation connects to a decade-old paper they haven't read
- Multiple sources discuss the same phenomenon without cross-referencing
- Hypotheses have supporting evidence scattered across time and sources
- Patterns emerge only when disparate information is connected

Traditional note-taking and document management fails here because it's optimized for storage and retrieval, not discovery.

## Solution

A knowledge graph where:
1. **Nodes** represent discrete units of knowledge (sources, observations, hypotheses, concepts, entities)
2. **Relationships** capture how these units connect (supports, contradicts, relates to)
3. **AI** continuously analyzes new data to suggest connections
4. **Feedback loops** train the system to better understand what connections matter

## Key Features

### Intelligent Connection Discovery
- When new data enters the system, LLM analyzes it against existing knowledge
- High-confidence connections are made automatically
- Uncertain connections are surfaced for human review
- System learns from user feedback to improve over time

### Interactive Graph Visualization
- Visual exploration of knowledge network
- Filter by node type, time range, confidence level
- Click to examine nodes and relationships
- Real-time updates as analysis completes

### Activity Feed
- Track what the system is doing
- Review pending connection suggestions
- Undo automated decisions
- See history of all changes

### Natural Language Queries
- Ask questions about your knowledge base
- "What evidence supports hypothesis X?"
- "What contradicts observation Y?"
- "Show me gaps in my understanding of entity Z"

## Target Users

- Academic researchers managing literature and observations
- Investigative journalists connecting sources and leads
- Analysts tracking patterns across data points
- Anyone building a body of knowledge over time

## Success Metrics

- Time to discover relevant connections (should decrease)
- Accuracy of suggested connections (should improve with feedback)
- User engagement with suggestions (approvals vs rejections)
- Novel insights surfaced (connections user wouldn't have found manually)

---

## Technical Approach

### Why Graph Database?

Traditional relational databases require knowing your queries upfront. Graph databases excel at:
- "Find everything connected to X within N hops"
- "What bridges these two clusters of information?"
- "Show me the path from observation A to hypothesis B"

These are exactly the queries that drive discovery.

### Why LLM Integration?

Embeddings and semantic similarity can find "similar" content, but understanding *why* two things relate (support vs contradiction vs tangential) requires reasoning. LLMs provide:
- Nuanced relationship classification
- Contextual understanding of domain
- Natural language explanation of connections
- Ability to recognize patterns humans describe

### Why Feedback Loop?

Every research domain has different standards for what constitutes a meaningful connection. A feedback loop allows:
- System adapts to user's research style
- Domain-specific patterns get reinforced
- False positive rate decreases over time
- System becomes collaborative tool, not just automation

---

## Project Boundaries

### In Scope
- Web-based graph visualization and management
- Automated connection suggestion with LLM
- User feedback collection and learning
- Real-time activity feed
- Basic natural language querying
- Multi-node type support (sources, observations, hypotheses, concepts, entities)

### Out of Scope (Future Considerations)
- Mobile application
- Multi-user collaboration features
- Advanced citation management
- PDF parsing and automatic source ingestion
- Integration with external databases (PubMed, arXiv)
- Formal ontology/reasoning engines

### Technical Constraints
- Python backend (team expertise)
- Must support both cloud LLM APIs and local models
- Should work on modest hardware (researcher laptop)
- Data privacy (option to keep everything local)
