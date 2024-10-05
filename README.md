# Isabelle/RL
This is the repository of the Isabelle/RL project. It is supported by the DeepIsaHOL MSCA Fellowship (number: 101102608) titled Reinforcement learning to improve proof-automation in theorem proving. The project's long-term objective is to use Isabelle as a reinforcement learning (RL) environment for training RL algorithms.

The project currently offers proof data retrieving capabilities using the [Isabelle proof assistant](https://isabelle.in.tum.de/) and the [scala-isabelle](https://github.com/dominique-unruh/scala-isabelle) library.

## Instalation

1. Prerequisites:
  * The [Isabelle2024](https://isabelle.in.tum.de/) proof assistant
  * The [scala-isabelle](https://github.com/dominique-unruh/scala-isabelle) library (see "Scala level" below)
  * The [sbt](https://www.scala-sbt.org/) build tool for Java and Scala
  * The [Py4J](https://www.py4j.org/install.html) library (see "Python level" below)
2. Pull this repository.
3. Adapt this project's `build.sbt` file to your needs (e.g. correct the location of `scala-isabelle`).
4. Adapt this project's `directories.scala` to your needs. Specifically, you will need to update the location of this project's `src` directory and paste it to `isabelle_rl`, and the location of your Isabelle application `isabelle_app`.
4. [Compile](https://www.scala-sbt.org/1.x/docs/Running.html) the project.
5. Set-up a [Python environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for your needs that includes the `Py4J` library.

## How it works?

You can retrieve data at three levels: Isabelle, Scala and Python. They are linearly ordered with Isabelle being the lowest level, and Python being the highest. The Isabelle proof assistant is in charge of retrieving all data from its sources. The other 2 levels interact with functions, structures and/or methods from the level immediately below. Each level satisfies a need that the previous level cannot. Scala enables a programmatic interaction with Isabelle while also having more features than the `Isabelle/ML` programming language. Python enables the possibility of interacting with popular machine learning, deep learning, and reinforcement learning libaries.

## How to use?

### Isabelle level
You can use Isabelle's jEdit PIDE to experiment with this project's `ML` libraries. For instance, this project's file `get.ML` includes various functions for retrieving specific data from the proof assistant. You can start a `.thy` file (e.g. `Temp.thy`) and see the result of `get.ML` functions by giving arguments to them inside `Temp.thy` and seeing the result in the PIDE's output panel:
```
theory "Temp"
imports "Complex_Main"

begin

ML_file "~/path/to/this/project/src/pred.ML"
ML_file "~/path/to/this/project/src/ops.ML"
ML_file "~/path/to/this/project/src/get.ML"

ML ‹Get.thms \<^context> "list_all2_eq"›

ML ‹Get.grouped_commands \<^theory>›

(* Returns a list of names and their corresponding list of theorems proved up to this point where the name contains the word "List" and the associated list has only one theorem *)
ML ‹val list_thms = Get.filtered_thms [on_fst (Get.passes_fact_check \<^context>), Pred.neg (on_snd Pred.has_many), on_fst (Pred.contains "List")] \<^context>›

end
```

### Scala level
This level is handled by `scala-isabelle`. The project provides a `minion.scala` that receives a working directory and manages Isabelle's theory stack via our `imports.scala` graph. The minion can use the `data.ML` function `Data.extract` to produce a string of `json`'s with proof data from a `.thy` file. Alternatively, a `writer.scala` uses the minion to retrieve the data and write it to a directory of your choice.

```scala
import isabelle_rl._
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._

val logic = "Ordinary_Differential_Equations"

val writer = Py4j_Gateway.get_writer(Directories.test_dir2, Directories.test_dir, logic)
  
val minion = writer.get_minion()

implicit val isabelle:de.unruh.isabelle.control.Isabelle = minion.isabelle

val jsons = minion.extract(Path.of("your/file.thy"))

writer.write_all()
```

### Python level
Finally, to exemplify the interaction with Python, the project provides a Python class in `writer.py` with the same functionality as `writer.scala` using itself a `minion.scala` (via `Py4j`). It copies the structure of a directory holding an Isabelle project and writes the `json`s of the project's proof data into another directory with the same structure as the original one:
```python
import sys
sys.path.append('/path/to/this/projects/src/main/python')
from writer import Writer

writer = Writer('/your/read/directory/with/an/isabelle/project/', '/your/write/directory/', 'the_logic_of_the_isabelle_projects_root_file')

writer.write_all()
```