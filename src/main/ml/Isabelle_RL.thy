theory Isabelle_RL
  imports Pure
  keywords "llm_recommend" :: diag
    and "show_proof_at" :: diag
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

ML_file "data.ML"
ML_file "json_maker.ML"
ML_file "g2tac_formatter.ML"
ML_file "writer.ML"

ML \<open>
val _ =
  Outer_Syntax.command \<^command_keyword>\<open>show_proof_at\<close>
    "quries a recommendation from a running llm"
    ((Parse.path -- Parse.int) >> (fn (file_str, line_num) => 
        Toplevel.keep (
          fn _ => 
          let
            val prf_data = Json_Maker.llm_proof_data file_str line_num;
          in Output.writeln prf_data end))
    );

val _ =
  Outer_Syntax.command \<^command_keyword>\<open>llm_recommend\<close>
    "quries a recommendation from a running llm"
    ((Parse.path -- Parse.int) 
      >> (fn (file_str, num) => 
        Toplevel.keep_proof (
          fn _ => 
          let
            val _ = Client.connect_to_server 5006;
            val prf_data = Json_Maker.llm_proof_data file_str num;
            val _ = Client.communicate prf_data;
            val response = Client.get_last_response ();
            val _ = Client.disconnect ();
            val _ = (case response of
              SOME sugg => Output.information (Active.sendback_markup_properties [] sugg)
              | NONE => Output.writeln "NONE suggestion from LLM")
          in () end))
    );
\<close>

end