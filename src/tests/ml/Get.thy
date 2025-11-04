theory Get
imports "Complex_Main"

begin

declare [[ML_print_depth = 4000000]]
ML_file "~/Programs/deepIsaHOL/src/main/ml/pred.ML"
ML_file "~/Programs/deepIsaHOL/src/main/ml/ops.ML"
ML_file "~/Programs/deepIsaHOL/src/main/ml/print.ML"
ML_file "~/Programs/deepIsaHOL/src/main/ml/imports.ML"
ML_file "~/Programs/deepIsaHOL/src/main/ml/get.ML"


lemma "\<forall>x. P x \<longrightarrow> P c"
  ML_val \<open>Get.user_state {break_lines=false} @{Isar.state}\<close>
  oops

ML \<open>Get.deps \<^theory> [@{thm list_induct2'}]\<close>


end