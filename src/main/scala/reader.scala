/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Reads proof data from theory files in the working directory
*/

package isabelle_rl
import scala.jdk.CollectionConverters._
import java.nio.file.{Files, Paths, Path}
import scala.util.Using
import de.unruh.isabelle.control.{Isabelle}
import de.unruh.isabelle.mlvalue.MLValue.{compileValue, compileFunction, compileFunction0}
import de.unruh.isabelle.mlvalue.{MLValue, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import de.unruh.isabelle.mlvalue.{StringConverter, UnitConverter}
import de.unruh.isabelle.pure.{Context, Theory}
import isabelle_rl.{Directories}

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._

object Reader {
  def apply(logic: String, work_dir: String): Reader = new Reader(logic, work_dir)
}

class Reader (val logic: String, val read_dir: String) {
  val setup: Isabelle.Setup = Isabelle.Setup(isabelleHome = Path.of(Directories.isabelle_app),
    sessionRoots = Nil,
    logic = logic,
    workingDirectory = Path.of(read_dir)
  )
  implicit val isabelle: Isabelle = new Isabelle(setup)
  val imports = Imports(read_dir)(isabelle)

  override def toString(): String = {
    "Reader(logic=" + logic + ", read_dir=" + read_dir + ")"
  }

  private object ML_Functions {
    final val isa_rl_thy_file = Directories.isabelle_rl + "Isabelle_RL.thy"
    val isabelle_rl_thy : Theory = Theory(Path.of(isa_rl_thy_file))
    val isa_rl_data = isabelle_rl_thy.importMLStructureNow("Data")
    final val extract : MLFunction2[Theory, String, String] = compileFunction[Theory, String, String](s"${isa_rl_data}.extract")
  }

  def extract (thy_file: String): List[String] = {
    // well-formed input
    val path = Paths.get(thy_file)
    val full_thy_file_path =
      if (path.isAbsolute) path 
      else Paths.get(this.read_dir, thy_file)
    if (!Files.exists(full_thy_file_path) || !Files.isRegularFile(full_thy_file_path)) {
      throw new IllegalArgumentException(s"Reader.extract: File not found or is not a file: $full_thy_file_path")
    }

    // theory processing
    val thy0 = imports.get_start_theory(full_thy_file_path)
    val thy_text = imports.get_file_text(full_thy_file_path)

    // extract data
    val jsons = ML_Functions.extract(thy0, thy_text)
    jsons.retrieveNow.split(" ISA_RL_SEP ").toList
  }

}