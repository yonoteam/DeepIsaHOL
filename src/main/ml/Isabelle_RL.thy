theory Isabelle_RL
  imports Pure
  keywords "llm_recommend" :: diag
begin

ML_file "pred.ML"
ML_file "ops.ML"
ML_file "imports.ML"
ML_file "get.ML"
ML_file "print.ML"

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
  Outer_Syntax.command \<^command_keyword>\<open>llm_recommend\<close>
    "quries a recommendation from a running llm"
    ((Resources.parse_file -- Parse.int) 
      >> (fn (get_file, num) => 
        Toplevel.keep_proof (
          fn state => 
          let
            val _ = Client.connect_to_server 5006;
            val thy = Toplevel.theory_of state;
            val file_path = Path.implode (#src_path (get_file thy));
            val prf_data = Json_Maker.llm_proof_data file_path state num;
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