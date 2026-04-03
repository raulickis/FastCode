"""
Graph Builder - Build code relationship graphs and export to Neo4j
"""

import os
import pickle
import logging
from typing import Dict, List, Any, Set, Optional
import networkx as nx
import tqdm
from neo4j import GraphDatabase

from .indexer import CodeElement
from .module_resolver import ModuleResolver
from .path_utils import file_path_to_module_path
from .symbol_resolver import SymbolResolver
from .call_extractor import CallExtractor
from .utils import ensure_dir


class CodeGraphBuilder:
    """Build various code relationship graphs and persist to Neo4j"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph_config = config.get("graph", {})
        self.logger = logging.getLogger(__name__)

        self.build_call_graph = self.graph_config.get("build_call_graph", True)
        self.build_dependency_graph = self.graph_config.get("build_dependency_graph", True)
        self.build_inheritance_graph = self.graph_config.get("build_inheritance_graph", True)
        self.max_depth = self.graph_config.get("max_depth", 5)

        # Graphs
        self.call_graph = nx.DiGraph()
        self.dependency_graph = nx.DiGraph()
        self.inheritance_graph = nx.DiGraph()

        # Maps for quick lookup
        self.element_by_name: Dict[str, CodeElement] = {}
        self.element_by_id: Dict[str, CodeElement] = {}
        self.imports_by_file: Dict[str, List[Dict]] = {}

        # Persistence
        self.persist_dir = config.get("vector_store", {}).get("persist_directory", "./data/vector_store")
        ensure_dir(self.persist_dir)

        # Neo4j Setup
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        try:
            self.neo4j_driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            self.logger.info(f"Neo4j driver initialized successfully connecting to {self.neo4j_uri}")
        except Exception as e:
            self.logger.warning(f"Neo4j driver initialization failed: {e}. Graph database export will be skipped.")
            self.neo4j_driver = None

    def build_graphs(self, elements: List[CodeElement], module_resolver: Optional[ModuleResolver] = None, symbol_resolver: Optional[SymbolResolver] = None):
        """
        Build all configured graphs

        Args:
            elements: List of code elements
            module_resolver: Optional ModuleResolver for precise dependency resolution
            symbol_resolver: Optional SymbolResolver for precise inheritance resolution
        """
        self.logger.info("Building code relationship graphs")

        # --- OPTIMIZATION: Pre-compute Lookup Maps ---
        # 1. Scope Lookup for Call Graph (replaces O(N) scan in _get_caller_id_from_scope)
        # Key: (file_path, type, name) -> element_id
        self.scope_lookup: Dict[tuple, str] = {}

        # 2. Class Lookup for Inheritance (replaces map building in fallback)
        # Key: class_name -> List[CodeElement]
        self.classes_by_name_lookup: Dict[str, List[CodeElement]] = {}

        # Index elements by name
        for elem in elements:
            self.element_by_name[elem.name] = elem
            self.element_by_id[elem.id] = elem

            # Populate Scope Lookup
            if elem.type in ["function", "class", "method"]:
                key = (elem.file_path, elem.type, elem.name)
                self.scope_lookup[key] = elem.id

            # Populate Class Lookup
            if elem.type == "class":
                if elem.name not in self.classes_by_name_lookup:
                    self.classes_by_name_lookup[elem.name] = []
                self.classes_by_name_lookup[elem.name].append(elem)

            # --- FIX: Selective Node Addition (Typing Check) ---
            # Only add nodes to graphs where they semantically belong.

            # 1. Dependency Graph: Only files
            if self.build_dependency_graph:
                if elem.type == "file":
                    self.dependency_graph.add_node(elem.id)

            # 2. Inheritance Graph: Only classes
            if self.build_inheritance_graph:
                if elem.type == "class":
                    self.inheritance_graph.add_node(elem.id)

            # 3. Call Graph: Functions, Methods, Classes
            if self.build_call_graph:
                if elem.type in ["function", "method", "class"]:
                    self.call_graph.add_node(elem.id)

            # Track imports
            if elem.type == "file":
                imports = elem.metadata.get("imports", [])
                if imports:
                    self.imports_by_file[elem.file_path] = imports

            # --- ADD LOGGING HERE ---
            ## <debug> with verify_shadowing.py
            if elem.name == "action":
                preprocessed_metadata = {k: v for k, v in elem.metadata.items() if k != 'embedding'}
                preprocessed_metadata['embedding'] = ' array([...])' if 'embedding' in elem.metadata else 'None'
                self.logger.debug(f"[DEBUG GRAPH] Found 'action' element. ID: {elem.id}, Type: {elem.type}")
                self.logger.debug(f"              Metadata: {preprocessed_metadata}")

        # Build graphs
        if self.build_dependency_graph:
            self._build_dependency_graph(elements, module_resolver)

        if self.build_inheritance_graph:
            self._build_inheritance_graph(elements, symbol_resolver)

        if self.build_call_graph:
            self._build_call_graph(elements, symbol_resolver)

        self.logger.info(
            f"Built graphs: "
            f"dependency ({self.dependency_graph.number_of_nodes()} nodes), "
            f"inheritance ({self.inheritance_graph.number_of_nodes()} nodes), "
            f"call ({self.call_graph.number_of_nodes()} nodes)"
        )

    def _build_dependency_graph(self, elements: List[CodeElement], module_resolver: Optional[ModuleResolver] = None):
        """
        Build file dependency graph based on imports using precise module resolution.
        """
        repo_root = self.config.get("repo_root", "")

        for elem in elements:
            if elem.type == "file":
                imports = elem.metadata.get("imports", [])

                # Get current file's module path
                current_module_path = self._get_module_path_from_file_id(elem.id, elements, repo_root)
                if not current_module_path:
                    continue

                # --- NEW: Check if this is a package file ---
                is_package = elem.file_path.endswith("__init__.py")

                for imp in imports:
                    module = imp.get("module", "")
                    names = imp.get("names", [])
                    level = imp.get("level", 0)

                    # Determine which modules to resolve
                    modules_to_resolve = []

                    if module:
                        # Case 1: Standard import - from module import names
                        modules_to_resolve.append(module)
                    elif level > 0 and names:
                        # Case 2: Relative import - from . import X, Y
                        modules_to_resolve.extend(names)

                    # Skip if nothing to resolve
                    if not modules_to_resolve:
                        continue

                    # Use ModuleResolver for precise resolution if available
                    if module_resolver:
                        for target_module in modules_to_resolve:
                            target_file_id = module_resolver.resolve_import(
                                current_module_path=current_module_path,
                                import_name=target_module,
                                level=level,
                                is_package=is_package
                            )

                            # Add edge if resolution succeeded
                            if target_file_id:
                                if target_file_id == elem.id:
                                    continue

                                # --- GAP 4 FIX: Cross-Repo Blocker Removed Here ---
                                # Previous code: if target_elem.repo_name != elem.repo_name: continue
                                # Now GraphRAG can link files across different microservices!

                                self.dependency_graph.add_edge(
                                    elem.id,
                                    target_file_id,
                                    type="imports",
                                    module=target_module,
                                    level=level,
                                    resolution_method="AST ModuleResolver"
                                )
                    else:
                        # Fallback to original logic
                        self.logger.warning(
                            f"ModuleResolver not provided for {elem.id}. "
                            f"Using fallback string matching - may produce false positives."
                        )
                        for target_module in modules_to_resolve:
                            for other_elem in elements:
                                # --- GAP 4 FIX: Cross-Repo Blocker Removed Here ---
                                # Previous code: if other_elem.repo_name != elem.repo_name: continue

                                if elem.id == other_elem.id:
                                    continue

                                if other_elem.type == "file" and target_module in other_elem.relative_path:
                                    self.dependency_graph.add_edge(
                                        elem.id,
                                        other_elem.id,
                                        type="imports",
                                        module=target_module,
                                        level=level,
                                        resolution_method="fallback_string_matching"
                                    )

    def _build_inheritance_graph(self, elements: List[CodeElement], symbol_resolver: Optional[SymbolResolver] = None):
        """
        Build class inheritance graph using precise symbol resolution
        """
        # Build inheritance relationships
        for elem in elements:
            if elem.type == "class":
                bases = elem.metadata.get("bases", [])

                for base_name in bases:
                    # Use SymbolResolver for precise resolution if available
                    if symbol_resolver:
                        # Get context needed for SymbolResolver
                        file_imports = self.imports_by_file.get(elem.file_path, [])
                        current_file_id = self._get_file_id_for_class_element(elem, elements)

                        if current_file_id:
                            # Resolve parent class using SymbolResolver
                            parent_class_id = symbol_resolver.resolve_symbol(
                                symbol_name=base_name,
                                current_file_id=current_file_id,
                                imports=file_imports
                            )

                            # Add edge if resolution succeeded
                            if parent_class_id:
                                self.inheritance_graph.add_edge(
                                    elem.id,
                                    parent_class_id,
                                    type="inherits",
                                    base_name=base_name
                                )
                                self.logger.debug(
                                    f"Added inheritance edge: {elem.id} -> {parent_class_id} "
                                    f"(resolved from '{base_name}')"
                                )
                            else:
                                self.logger.debug(
                                    f"Could not resolve parent class '{base_name}' for {elem.id}"
                                )
                        else:
                            self.logger.warning(
                                f"Could not determine file_id for class {elem.id}, "
                                f"falling back to local resolution"
                            )
                            self._fallback_to_local_inheritance_resolution(elem, base_name, elements)
                    else:
                        self.logger.warning(
                            f"SymbolResolver not provided for {elem.id}. "
                            f"Using fallback local name matching - may miss cross-file inheritance."
                        )
                        self._fallback_to_local_inheritance_resolution(elem, base_name, elements)

    def _get_file_id_for_class_element(self, class_elem: CodeElement, elements: List[CodeElement]) -> Optional[str]:
        """
        Get file ID for a class element by finding the corresponding file element
        """
        for elem in elements:
            if elem.type == "file" and elem.file_path == class_elem.file_path:
                return elem.id

        if class_elem.file_path:
            import os
            filename = os.path.basename(class_elem.file_path)
            name, _ = os.path.splitext(filename)
            return f"file_{name}"

        return None

    def _fallback_to_local_inheritance_resolution(self, elem: CodeElement, base_name: str, elements: List[CodeElement]):
        """
        Fallback method for local-only inheritance resolution.
        """
        candidates = self.classes_by_name_lookup.get(base_name, [])

        if not candidates:
            return

        # --- GAP 4 FIX: Remove artificial scoping prioritizing same repository ---
        # We now accept cross-repo inheritance matching if names match.
        best_match = candidates[0]

        if best_match:
            self.inheritance_graph.add_edge(
                elem.id,
                best_match.id,
                type="inherits",
                base_name=base_name,
                resolution_method="fallback_local_matching_cross_repo"
            )
            self.logger.debug(
                f"Added inheritance edge via fallback: {elem.id} -> {best_match.id} "
                f"(matched '{base_name}' in repo '{best_match.repo_name}')"
            )

    def _build_call_graph(self, elements: List[CodeElement], symbol_resolver: Optional[SymbolResolver] = None):
        """
        Build function call graph using CallExtractor and SymbolResolver (Task 4.4)

        Args:
            elements: List of code elements
            symbol_resolver: SymbolResolver for resolving callee definitions
        """
        if not symbol_resolver:
            self.logger.warning(
                "SymbolResolver not provided for call graph building. "
                "Skipping call graph construction."
            )
            return

        # Initialize CallExtractor
        call_extractor = CallExtractor()

        # Statistics
        total_calls = 0
        linked_calls = 0

        pbar_elements = tqdm.tqdm(elements, desc=f"Building call graph")
        for elem in pbar_elements:
            if elem.type == "file":
                # Extract calls from this file (Task 4.3)
                calls = call_extractor.extract_calls(elem.code, elem.file_path)
                total_calls += len(calls)

                # Extract instance variable types for enhanced resolution (Phase 1)
                file_instance_types = call_extractor.extract_instance_types(elem.code)

                # --- ADD DEBUG LOGGING HERE ---
                if file_instance_types:
                    self.logger.debug(f"[EXTRACT] Instance types found in {elem.file_path}:")
                    for scope, vars_map in file_instance_types.items():
                        self.logger.debug(f"    Scope '{scope}': {list(vars_map.keys())}")
                        # Detailed log for __init__ methods to debug Bug #8
                        if "init" in scope or "class" in scope:
                            self.logger.debug(f"        -> Vars: {vars_map}")
                # -----------------------------

                # Get imports context for this file
                file_imports = self.imports_by_file.get(elem.file_path, [])

                for call in calls:
                    # Determine caller ID (using scope_id from Task 4.3)
                    caller_id = self._get_caller_id_from_scope(call, elem, elements)

                    # Retrieve the actual caller element to get its class context
                    caller_elem = self.element_by_id.get(caller_id) if caller_id else None

                    # Determine callee ID using SymbolResolver with instance type inference (Phase 2)
                    callee_ids = self._resolve_callee_with_symbol_resolver(
                        call, elem.id, file_imports, symbol_resolver, file_instance_types,
                        caller_elem=caller_elem
                    )

                    # Add edge(s) for each resolved callee (now supports one-to-many)
                    if caller_id and callee_ids:
                        for callee_id in callee_ids:
                            self.call_graph.add_edge(
                                caller_id,
                                callee_id,
                                type="calls",
                                call_name=call['call_name'],
                                call_type=call.get('call_type', 'unknown'),
                                file_path=call['file_path'],
                                node_text=call.get('node_text', '')
                            )
                            linked_calls += 1

                            self.logger.debug(
                                f"Added call edge: {caller_id} -> {callee_id} "
                                f"('{call['call_name']}' in {call.get('call_type', 'unknown')} call)"
                            )
                    else:
                        self.logger.debug(
                            f"Could not link call: '{call['call_name']}' "
                            f"(caller: {caller_id}, callee: {callee_ids})"
                        )

        self.logger.info(
            f"Call graph built: {linked_calls}/{total_calls} calls successfully linked "
            f"({linked_calls / total_calls * 100 if total_calls > 0 else 0:.1f}% success rate)"
        )

    def get_related_elements(self, element_id: str, max_hops: int = 2) -> Set[str]:
        """
        Get related elements within max_hops distance
        """
        related = set()

        # Check all graphs
        for graph in [self.dependency_graph, self.inheritance_graph, self.call_graph]:
            if element_id in graph:
                # Get predecessors and successors within max_hops
                for node in nx.single_source_shortest_path_length(
                        graph, element_id, cutoff=max_hops
                ).keys():
                    related.add(node)

                # Also check reverse direction
                for node in nx.single_source_shortest_path_length(
                        graph.reverse(), element_id, cutoff=max_hops
                ).keys():
                    related.add(node)

        return related

    def _get_module_path_from_file_id(self, file_id: str, elements: List[CodeElement], repo_root: str) -> Optional[str]:
        """
        Get module path from file ID by looking up the corresponding CodeElement.
        """
        for elem in elements:
            if elem.id == file_id and elem.type == "file":
                return file_path_to_module_path(elem.file_path, repo_root)

        return None

    def get_dependencies(self, element_id: str) -> List[str]:
        """Get direct dependencies of an element"""
        if element_id in self.dependency_graph:
            return list(self.dependency_graph.successors(element_id))
        return []

    def get_dependents(self, element_id: str) -> List[str]:
        """Get elements that depend on this element"""
        if element_id in self.dependency_graph:
            return list(self.dependency_graph.predecessors(element_id))
        return []

    def get_subclasses(self, element_id: str) -> List[str]:
        """Get subclasses of a class"""
        if element_id in self.inheritance_graph:
            return list(self.inheritance_graph.predecessors(element_id))
        return []

    def get_superclasses(self, element_id: str) -> List[str]:
        """Get superclasses of a class"""
        if element_id in self.inheritance_graph:
            return list(self.inheritance_graph.successors(element_id))
        return []

    def get_callers(self, element_id: str) -> List[str]:
        """Get functions that call this function"""
        if element_id in self.call_graph:
            return list(self.call_graph.predecessors(element_id))
        return []

    def get_callees(self, element_id: str) -> List[str]:
        """Get functions called by this function"""
        if element_id in self.call_graph:
            return list(self.call_graph.successors(element_id))
        return []

    def find_path(self, source_id: str, target_id: str,
                  graph_type: str = "dependency") -> Optional[List[str]]:
        """
        Find path between two elements
        """
        graph_map = {
            "dependency": self.dependency_graph,
            "inheritance": self.inheritance_graph,
            "call": self.call_graph,
        }

        graph = graph_map.get(graph_type)
        if graph is None:
            return None

        try:
            return nx.shortest_path(graph, source_id, target_id)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graphs"""
        stats = {}

        for name, graph in [
            ("dependency", self.dependency_graph),
            ("inheritance", self.inheritance_graph),
            ("call", self.call_graph),
        ]:
            stats[name] = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(graph),
            }

            if graph.number_of_nodes() > 0:
                try:
                    stats[name]["avg_degree"] = sum(
                        d for _, d in graph.degree()
                    ) / graph.number_of_nodes()
                except:
                    stats[name]["avg_degree"] = 0

        return stats

    def save(self, name: str = "index"):
        """
        Save graph data to disk and export to Neo4j database
        """
        graph_path = os.path.join(self.persist_dir, f"{name}_graphs.pkl")

        try:
            with open(graph_path, 'wb') as f:
                pickle.dump({
                    "call_graph": self.call_graph,
                    "dependency_graph": self.dependency_graph,
                    "inheritance_graph": self.inheritance_graph,
                    "element_by_name": {k: v.to_dict() for k, v in self.element_by_name.items()},
                    "element_by_id": {k: v.to_dict() for k, v in self.element_by_id.items()},
                    "imports_by_file": self.imports_by_file,
                }, f)

            self.logger.info(f"Saved graph data to {graph_path}")

            # Neo4j Export triggered here
            if self.neo4j_driver:
                self._export_to_neo4j(name)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save graph data: {e}")
            return False

    def _export_to_neo4j(self, repo_name: str):
        """Export nodes and edges to Neo4j database using batch execution"""
        self.logger.info(f"Exporting graph data to Neo4j for {repo_name}...")

        nodes_data = []
        for elem_id, elem in self.element_by_id.items():
            nodes_data.append({
                "id": elem_id,
                "name": elem.name,
                "type": elem.type,
                "repo_name": elem.repo_name or "unknown",
                "file_path": elem.relative_path
            })

        dependency_edges = []
        for u, v, data in self.dependency_graph.edges(data=True):
            dependency_edges.append({"source": u, "target": v, "module": data.get("module", "")})

        inheritance_edges = []
        for u, v, data in self.inheritance_graph.edges(data=True):
            inheritance_edges.append({"source": u, "target": v, "base_name": data.get("base_name", "")})

        call_edges = []
        for u, v, data in self.call_graph.edges(data=True):
            call_edges.append({"source": u, "target": v, "call_name": data.get("call_name", "")})

        def _execute_batch(tx):
            # Upsert Nodes
            tx.run("""
                UNWIND $nodes AS node
                MERGE (e:CodeElement {id: node.id})
                SET e.name = node.name,
                    e.type = node.type,
                    e.repo_name = node.repo_name,
                    e.file_path = node.file_path
            """, nodes=nodes_data)

            # Upsert Dependency Edges
            if dependency_edges:
                tx.run("""
                    UNWIND $edges AS edge
                    MATCH (source:CodeElement {id: edge.source})
                    MATCH (target:CodeElement {id: edge.target})
                    MERGE (source)-[r:DEPENDS_ON]->(target)
                    SET r.module = edge.module
                """, edges=dependency_edges)

            # Upsert Inheritance Edges
            if inheritance_edges:
                tx.run("""
                    UNWIND $edges AS edge
                    MATCH (source:CodeElement {id: edge.source})
                    MATCH (target:CodeElement {id: edge.target})
                    MERGE (source)-[r:INHERITS]->(target)
                    SET r.base_name = edge.base_name
                """, edges=inheritance_edges)

            # Upsert Call Edges
            if call_edges:
                tx.run("""
                    UNWIND $edges AS edge
                    MATCH (source:CodeElement {id: edge.source})
                    MATCH (target:CodeElement {id: edge.target})
                    MERGE (source)-[r:CALLS]->(target)
                    SET r.call_name = edge.call_name
                """, edges=call_edges)

        try:
            with self.neo4j_driver.session() as session:
                session.execute_write(_execute_batch)
            self.logger.info(f"Successfully exported graph to Neo4j for {repo_name}.")
        except Exception as e:
            self.logger.error(f"Failed to export to Neo4j: {e}")

    def load(self, name: str = "index") -> bool:
        """
        Load graph data from disk
        """
        graph_path = os.path.join(self.persist_dir, f"{name}_graphs.pkl")

        if not os.path.exists(graph_path):
            self.logger.warning(f"Graph data not found: {graph_path}")
            return False

        try:
            with open(graph_path, 'rb') as f:
                data = pickle.load(f)

                # --- DEBUG LOGS: Check for data loss due to name collisions ---
                saved_by_name = len(data.get("element_by_name", {}))
                saved_by_id = len(data.get("element_by_id", {}))
                self.logger.info(f"[DEBUG] Load Stats - Saved by Name: {saved_by_name}, Saved by ID: {saved_by_id}")

                if saved_by_id > saved_by_name:
                    self.logger.warning(f"[DEBUG] BUG DETECTED: Lost {saved_by_id - saved_by_name} elements due to name collisions!")
                # ------------------------------------------------------------

                self.call_graph = data["call_graph"]
                self.dependency_graph = data["dependency_graph"]
                self.inheritance_graph = data["inheritance_graph"]
                self.imports_by_file = data["imports_by_file"]

                # Reconstruct indices with CodeElement objects
                from .indexer import CodeElement
                self.element_by_name = {}
                self.element_by_id = {}

                # --- FIX: Use element_by_id to avoid data loss from duplicate names ---
                if "element_by_id" in data:
                    source_data = data["element_by_id"]
                    self.logger.info(f"Restoring {len(source_data)} elements from unique ID index.")
                else:
                    self.logger.warning("Legacy cache detected: restoring from element_by_name (some duplicate functions may be lost).")
                    source_data = data["element_by_name"]

                for k, v in source_data.items():
                    elem = CodeElement(**v)
                    self.element_by_id[elem.id] = elem
                    self.element_by_name[elem.name] = elem
                # -------------------------------------------------------------------

            self.logger.info(
                f"Loaded graph data with "
                f"{self.dependency_graph.number_of_nodes()} dependency nodes, "
                f"{self.inheritance_graph.number_of_nodes()} inheritance nodes, "
                f"{self.call_graph.number_of_nodes()} call nodes"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to load graph data: {e}")
            return False

    def merge_from_file(self, name: str) -> bool:
        """
        Load and merge graph data from another repository into current graphs
        """
        graph_path = os.path.join(self.persist_dir, f"{name}_graphs.pkl")

        if not os.path.exists(graph_path):
            self.logger.warning(f"Graph data not found for merging: {graph_path}")
            return False

        try:
            with open(graph_path, 'rb') as f:
                data = pickle.load(f)
                other_call_graph = data["call_graph"]
                other_dependency_graph = data["dependency_graph"]
                other_inheritance_graph = data["inheritance_graph"]
                other_imports_by_file = data["imports_by_file"]

                # --- FIX: Use element_by_id to avoid data loss from duplicate names ---
                if "element_by_id" in data:
                    other_elements = data["element_by_id"]
                    self.logger.info(f"Merging {len(other_elements)} elements from unique ID index.")
                else:
                    self.logger.warning("Legacy cache detected in merge: using element_by_name (some duplicate functions may be lost).")
                    other_elements = data["element_by_name"]
                # -------------------------------------------------------------------

            # Merge graphs using NetworkX compose
            self.call_graph = nx.compose(self.call_graph, other_call_graph)
            self.dependency_graph = nx.compose(self.dependency_graph, other_dependency_graph)
            self.inheritance_graph = nx.compose(self.inheritance_graph, other_inheritance_graph)

            # Merge elements from source file
            from .indexer import CodeElement
            for v in other_elements.values():
                elem = CodeElement(**v)
                if elem.id not in self.element_by_id:
                    self.element_by_name[elem.name] = elem
                    self.element_by_id[elem.id] = elem

            self.imports_by_file.update(other_imports_by_file)

            self.logger.info(
                f"Merged graph data from {name}. Total graphs now: "
                f"{self.dependency_graph.number_of_nodes()} dependency nodes, "
                f"{self.inheritance_graph.number_of_nodes()} inheritance nodes, "
                f"{self.call_graph.number_of_nodes()} call nodes"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to merge graph data from {name}: {e}")
            return False

    def _get_caller_id_from_scope(self, call: Dict[str, Any], file_elem: CodeElement, elements: List[CodeElement]) -> Optional[str]:
        """
        Get caller ID from call scope information (Task 4.4)
        OPTIMIZED: O(1) lookup instead of O(N) scan
        """
        scope_id = call.get('scope_id')

        # --- ADD DEBUG LOGGING HERE ---
        # Only log if it's a method call we are interested in (reduce noise)
        if call.get('base_object') in ['self', 'cls'] or (scope_id and "class" in scope_id):
            self.logger.debug(f"[SCOPE] Processing call '{call.get('call_name')}' inside scope: '{scope_id}'")
        # -----------------------------

        if scope_id is None:
            return file_elem.id

        scope_parts = scope_id.split('::', 1)
        if len(scope_parts) != 2:
            self.logger.warning(f"Invalid scope_id format: {scope_id}")
            return file_elem.id

        scope_type, scope_name = scope_parts
        key = (file_elem.file_path, scope_type, scope_name)
        caller_id = self.scope_lookup.get(key)

        if not caller_id:
            self.logger.debug(
                f"Could not find {scope_type} element '{scope_name}' in {file_elem.file_path}"
            )

        return caller_id

    def _resolve_callee_with_symbol_resolver(self, call: Dict[str, Any], current_file_id: str,
                                             file_imports: List[Dict[str, Any]],
                                             symbol_resolver: SymbolResolver,
                                             file_instance_types: Optional[Dict[str, Dict[str, List[str]]]] = None,
                                             caller_elem: Optional[CodeElement] = None) -> List[str]:
        """
        Resolve callee definition using SymbolResolver with instance variable type inference.
        """
        call_name = call['call_name']
        call_type = call.get('call_type', 'simple')
        base_object = call.get('base_object')

        # Case 1: Simple function call: func()
        if call_type == 'simple':
            resolved_id = symbol_resolver.resolve_symbol(call_name, current_file_id, file_imports)
            return [resolved_id] if resolved_id else []

        # Case 2: Module function call OR Instance method call
        elif call_type == 'attribute' and base_object:

            # --- FIX: Check if base_object is a local variable first! ---
            call_scope = call.get('scope_id') or "global"
            is_local_var = False

            if file_instance_types:
                if base_object in file_instance_types.get(call_scope, {}):
                    is_local_var = True
                elif base_object in file_instance_types.get("global", {}):
                    is_local_var = True
                elif base_object in file_instance_types.get("function::__init__", {}):
                    is_local_var = True

            if not is_local_var:
                for import_info in file_imports:
                    if import_info.get('module') == base_object:
                        full_call_name = f"{base_object}.{call_name}"
                        resolved_id = symbol_resolver.resolve_symbol(full_call_name, current_file_id, file_imports)
                        return [resolved_id] if resolved_id else []

            # If not a module call, check if it's self/cls call
            if base_object in ['self', 'cls']:
                # Case 3: Method call: obj.method() or self.method()
                if caller_elem and caller_elem.type == 'function':
                    class_name = caller_elem.metadata.get('class_name')
                    if class_name:
                        full_method_name = f"{class_name}.{call_name}"
                        resolved_id = symbol_resolver.resolve_symbol(full_method_name, current_file_id, file_imports)
                        if resolved_id:
                            return [resolved_id]

                resolved_id = symbol_resolver.resolve_symbol(call_name, current_file_id, file_imports)
                return [resolved_id] if resolved_id else []
            else:
                # --- ADD DEBUG LOGGING HERE ---
                self.logger.debug(f"[ROUTING] Routing '{base_object}.{call_name}' to Instance Method Resolution")
                self.logger.debug(f"          Context: File={current_file_id}, Scope={call_scope}")
                # -----------------------------

                # Case 4: Instance method call
                return self._resolve_instance_method_call(
                    base_object, call_name, current_file_id, file_imports,
                    symbol_resolver, file_instance_types or {}, call_scope
                )

        # Default case: try simple resolution
        resolved_id = symbol_resolver.resolve_symbol(call_name, current_file_id, file_imports)
        return [resolved_id] if resolved_id else []

    def _resolve_instance_method_call(self, base_object: str, call_name: str,
                                      current_file_id: str, file_imports: List[Dict[str, Any]],
                                      symbol_resolver: SymbolResolver,
                                      file_instance_types: Dict[str, Dict[str, List[str]]],
                                      scope_id: str) -> List[str]:
        """
        Resolve instance method calls using type inference (Phase 2.3).
        """
        # --- DEBUG LOGGING START ---
        self.logger.debug(f"[RESOLVE] Attempting to resolve '{base_object}.{call_name}' from Scope: '{scope_id}'")
        # --- DEBUG LOGGING END ---

        # 1. Try to find variable in the current local scope
        local_types = file_instance_types.get(scope_id, {})
        candidate_classes = local_types.get(base_object)

        # --- DEBUG LOGGING START ---
        if candidate_classes:
            self.logger.debug(f"    [+] Found type in LOCAL scope '{scope_id}': {candidate_classes}")
        else:
            self.logger.debug(f"    [-] Not found in LOCAL scope '{scope_id}'")
        # --- DEBUG LOGGING END ---

        # 2. If not found locally, check '__init__' scope (Fix for Bug #8)
        if not candidate_classes:
            candidate_classes = file_instance_types.get("function::__init__", {}).get(base_object)

            if candidate_classes:
                self.logger.debug(f"    [+] Found type in __init__ scope: {candidate_classes}")

        # 3. If not found, check 'global' scope
        if not candidate_classes:
            candidate_classes = file_instance_types.get("global", {}).get(base_object)

            # --- DEBUG LOGGING START ---
            if candidate_classes:
                self.logger.debug(f"    [+] Found type in GLOBAL scope: {candidate_classes}")
            else:
                self.logger.debug(f"    [-] Not found in GLOBAL scope")

                found_in_other_scopes = []
                for s_id, vars_map in file_instance_types.items():
                    if base_object in vars_map:
                        found_in_other_scopes.append(f"{s_id} -> {vars_map[base_object]}")

                if found_in_other_scopes:
                    self.logger.debug(f"    [!] Variable '{base_object}' exists in other scopes: {found_in_other_scopes}")
            # --- DEBUG LOGGING END ---

        if not candidate_classes:
            return []

        resolved_ids = []

        # For each candidate class, try to resolve the method
        for class_name in candidate_classes:
            # Step 1: Resolve the class definition first
            class_id = symbol_resolver.resolve_symbol(class_name, current_file_id, file_imports)

            if not class_id:
                self.logger.debug(f"    [-] Could not resolve class definition for '{class_name}'")
                continue

            # Step 2: Try to resolve the method within that class
            full_method_name = f"{class_name}.{call_name}"
            method_id = symbol_resolver.resolve_symbol(full_method_name, current_file_id, file_imports)

            if method_id:
                resolved_ids.append(method_id)
                self.logger.debug(f"    [+] Resolved method '{full_method_name}' to {method_id}")
            else:
                # Fallback: Try resolving just the method name
                method_id = symbol_resolver.resolve_symbol(call_name, current_file_id, file_imports)

                if method_id:
                    resolved_ids.append(method_id)
                else:
                    # Final Fallback: Look inside the class file directly
                    class_elem = self.element_by_id.get(class_id)
                    found_in_class_file = False

                    if class_elem:
                        for elem in self.element_by_id.values():
                            if (elem.type == 'function' and
                                    elem.name == call_name and
                                    elem.file_path == class_elem.file_path and
                                    elem.metadata.get('class_name') == class_elem.name):
                                resolved_ids.append(elem.id)
                                self.logger.debug(f"    [+] Resolved method '{call_name}' in class '{class_name}' via file lookup")
                                found_in_class_file = True
                                break

                    if not found_in_class_file:
                        # Final Fallback: Link to the class itself (Recall over Precision)
                        resolved_ids.append(class_id)
                        self.logger.debug(
                            f"    [~] Could not resolve method '{call_name}' in class '{class_name}', "
                            f"linking to class {class_id} instead"
                        )

        # Deduplicate resolved IDs while preserving order
        seen = set()
        unique_resolved_ids = []
        for resolved_id in resolved_ids:
            if resolved_id not in seen:
                seen.add(resolved_id)
                unique_resolved_ids.append(resolved_id)

        return unique_resolved_ids

    def __del__(self):
        """Cleanup Neo4j connection when object is destroyed"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()
