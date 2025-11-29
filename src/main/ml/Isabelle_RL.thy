theory Isabelle_RL
  imports Pure
  keywords "llm_recommend" :: diag
    and "show_proof_at" :: diag
    and "llm_try_proof" :: diag
    and "llm_init" :: diag
begin

ML_file "pred.ML"
ML_file "ops.ML"
ML_file "print.ML"
ML_file "imports.ML"
ML_file "get.ML"

ML_file "sections.ML"
ML_file "seps.ML"
ML_file "actions.ML"

ML_file "repl_state.ML"
ML_file "client.ML"
ML_file "llm.ML"

ML_file "data.ML"
ML_file "json.ML"
ML_file "writer.ML"

ML \<open>
val _ =
  Outer_Syntax.command \<^command_keyword>\<open>llm_init\<close>
    "enables loading the right Isabelle context for the LLM"
    (Parse.path >> (fn file_str =>
      Toplevel.keep (fn _ =>
        let
          val file_path = Path.explode file_str;
          val _ = Imports.init_for file_path;
          val _ = Imports.load_upto_start_of file_path;
        in () end
      )
    ));

val _ =
  Outer_Syntax.command \<^command_keyword>\<open>show_proof_at\<close>
    "shows the proof progress up to the input line"
    ((Parse.int) >> (fn line_num => 
        Toplevel.keep (
          fn _ => 
          let
            val work_path = Imports.get_work_path ();
            val thy0 = Imports.start_thy_of work_path;
            val prf_data = LLM.json_text_at thy0 (Path.implode work_path) line_num;
          in Output.writeln prf_data end))
    );

val _ =
  Outer_Syntax.command \<^command_keyword>\<open>llm_recommend\<close>
    "quries a recommendation from a running llm"
    ((Parse.int) 
      >> (fn num => 
        Toplevel.keep_proof (
          fn _ => 
          let
            val work_path = Imports.get_work_path ();
            val thy0 = Imports.start_thy_of work_path;
            val _ = Client.connect_to_server 5006;
            val prf_data = LLM.json_text_at thy0 (Path.implode work_path) num;
            val _ = Client.communicate prf_data;
            val response = Client.get_last_response ();
            val _ = Client.disconnect ();
            val _ = (case response of
              SOME sugg => Output.information (Active.sendback_markup_properties [] sugg)
              | NONE => Output.writeln "NONE suggestion from LLM")
          in () end))
    );

val _ =
  Outer_Syntax.command \<^command_keyword>\<open>llm_try_proof\<close>
    "attempts a proof by repeatedly calling a running llm"
    ((Parse.int) 
      >> (fn num => 
        Toplevel.keep_proof (
          fn _ => 
          let
            val work_path = Imports.get_work_path ();
            val thy0 = Imports.start_thy_of work_path;
            val _ = Client.connect_to_server 5006;
            val prf_data = Actions.proof_at thy0 (Path.implode work_path) num;
            val prfs = LLM.prove 3 prf_data;
            val _ = Client.disconnect ();
            val _ = map (Output.information o Active.sendback_markup_properties []) prfs;
          in () end))
    );
\<close>

end