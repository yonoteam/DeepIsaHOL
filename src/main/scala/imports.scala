/*
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz
 */

package isabelle_rl

import java.nio.file.{Path, Paths, Files}
import java.io.FileNotFoundException
import scala.jdk.CollectionConverters._
import de.unruh.isabelle.control.{Isabelle, OperationCollection}
import de.unruh.isabelle.mlvalue.MLValue.{compileFunction, compileFunction0}
import de.unruh.isabelle.pure.{Position, Theory, TheoryHeader, ToplevelState}
import de.unruh.isabelle.mlvalue.{AdHocConverter, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import isabelle_rl.Theory_Loader.{Heap, Source, Text}
import isabelle_rl.Theory_Loader.Ops
import isabelle_rl.Graph

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._
import scala.concurrent.ExecutionContext.Implicits.global

class Imports (val work_dir: Path)(implicit isabelle: Isabelle) {
  val local_thy_files: List[Path] = {
    val files = Files.walk(work_dir).iterator().asScala
    val filtered_files = files.filter { path => Files.isRegularFile(path) && path.toString.endsWith(".thy")}
    filtered_files.toList
  }
  
  private def file_name_without_extension(file_path: Path): String = {
    val file_name = file_path.getFileName.toString
    if (file_name.contains(".")) {
      file_name.substring(0, file_name.lastIndexOf('.'))
    } else {
      file_name
    }
  }

  def locate_via_thy(import_name: String): Option[Path] = {
    if (Ops.can_get_thy(import_name).retrieveNow) Some(Path.of("ISABELLE/" + import_name)) else None
  }

  def locate_locally(import_name: String): Option[Path] = {
    val file_name = if (import_name.contains(".")) {
      Ops.get_base_name(import_name).retrieveNow + ".thy"
    } else if (import_name.contains("/")) {
      Path.of(import_name + ".thy").getFileName.toString
    } else {
      import_name + ".thy"
    }
    local_thy_files.find(_.getFileName.toString == file_name) match {
      case Some(result_path) => return Some(result_path)
      case None => None
    }
  }

  def locate_via_thy_file(import_name: String): Option[Path] = {
    Ops.find_thy_file(import_name).retrieveNow
  }

  def locate(import_name: String): Path = {
    val file_path: Option[Path] = locate_locally(import_name)
      .orElse(locate_via_thy(import_name))
      .orElse(locate_via_thy_file(import_name))
    file_path match {
      case Some(path) => path
      case None => throw new Exception(s"Imports.locate could not find $import_name in $work_dir")
    }
  }

  def init_deps(debug: Boolean): Graph[Path, Option[Theory]] = {
    var file_dep_graph: Graph[Path, Option[Theory]] = Graph.empty

    local_thy_files.foreach { thy_file_path =>
      if (debug) println(s"Adding local ${thy_file_path.toString}")
      file_dep_graph = file_dep_graph.new_node(thy_file_path, None)
    }

    local_thy_files.foreach { thy_file_path =>
      if (debug) println(s"Processing parents of ${thy_file_path.toString}")
      val parents = Text.from_file(thy_file_path).get_imports.map(locate) 
      // TODO: we get the imports, if the import can be loaded via can_get_thy or can_get_thy_file add node (locate(import), Some(Theory(import)))
      parents.foreach{ parent =>
        if (debug) println(s"Processing parent ${parent.toString}")
        file_dep_graph = file_dep_graph.default_node(parent, None)
        file_dep_graph = file_dep_graph.add_edge(thy_file_path, parent)
      }
    }

    file_dep_graph
  }
}

/*
def load_deps(original: Graph[Path, Option[Theory]], debug: Boolean): Graph[Path, Option[Theory]] = {
    val eldests = original.maximals

    def update_eldests(original: Graph[Path, Option[Theory]]) {
    var updated = original
    eldests.foreach { elder =>
        if (elder.startsWith(Path.of("ISABELLE"))) {
        updated = updated.map_node(elder, (_ => Some(Theory(file_name_without_extension(elder)))))
        } else {
        throw Exception(s"Imports.load_deps input graph is  $import_name in $work_dir")
        }
    }
    updated
    }

    def update_node(thy_file_path: Path, graph: Graph[Path, Option[Theory]]): Graph[Path, Option[Theory]] = {
    // Get the parent nodes and retrieve their theories
    val parent_paths = graph.imm_succs(thy_file_path).toList
    val parent_theories = parent_paths.map(graph.get_node) match {
        case theories if theories.forall(_.isDefined) => theories.flatten // Get all Some(theory)
        case _ => throw new Exception(s"Missing theory for one of the parents of $thy_file_path")
    }
    // Compute the current theory using Ops.begin_theory
    val thy_header = get_header(thy_file_path) // Assuming this is a provided function
    val master_dir = thy_file_path.getParent
    val current_theory = Ops.begin_theory(master_dir, thy_header, parent_theories).retrieveNow
    
    // Update the graph with the computed theory for this node
    graph.map_entry(thy_file_path, {
        case (_, (preds, succs)) => (Some(current_theory), (preds, succs))
    })
    }
    val all_nodes = original.all_preds(eldests)
} } */ 