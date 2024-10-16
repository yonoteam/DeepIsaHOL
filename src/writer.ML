
signature WRITER =
sig
  (* G2TAC *)
  val write_g2tac_proof: string -> (Data.env * Data.T Sections.T) -> unit
  val write_g2tac_proofs: (string * theory * string) -> unit

  (* JSONS *)
  val extract_jsons: theory * string -> string
  val write_json_proof: string -> (Data.env * Data.T Sections.T) -> unit
  val write_json_proofs: (string * theory * string) -> unit
end;


structure Writer: WRITER =
struct

(* val _ = ML_system_pp (fn _ => fn _ 
  => Pretty.to_polyml o pretty_json true 0 (Context.the_local_context ())) *)


(** G2TAC **)

(* write_g2tac_proof: string -> (Data.env * Data.T Sections.T) -> unit *)
fun write_g2tac_proof write_file_path (env, proof) =
  let
    val steps = Sections.recover_trace proof
    fun write_next step () =
      let
        val backslash = "\\";
        val start_quote = backslash ^ "<";
        val to_write = G2tac_Formatter.pretty_step env step
          |> Print.string_of_pretty
          |> Ops.replace_all start_quote (backslash ^ start_quote)
      in Print.write_to (Print.FILE write_file_path) to_write end;
  in fold write_next steps () end;

(* write_g2tac_proof': TextIO.outstream 
  -> Data.env * Data.T Sections.T 
  -> unit Exn.result list *)
fun write_g2tac_proof' file_stream (env, proof) =
  let
    val steps = Sections.recover_trace proof
    fun write_next step errs =
      let
        val backslash = "\\";
        val start_quote = backslash ^ "<";
        val to_write = 
          G2tac_Formatter.pretty_step env step
          |> Print.string_of_pretty
          |> Ops.replace_all start_quote (backslash ^ start_quote)
      in Exn.result TextIO.output (file_stream, to_write) :: errs end;
  in fold write_next steps [] end;

(* write_g2tac_proofs: (string * theory * string) -> unit *)
fun write_g2tac_proofs (write_file_path, thy0, thy_text) =
  let 
    val proofs = Data.extract_proofs thy0 thy_text;
    val file_stream = TextIO.openAppend(write_file_path);
    fun write_next proof errs =
      (write_g2tac_proof' file_stream proof) @ errs
    val errors = fold write_next proofs [];
    val _ = TextIO.closeOut(file_stream)
    val _ = Par_Exn.release_all (rev errors)
  in () end;


(** JSONS **)

(* extract_jsons: theory * string -> string *)
fun extract_jsons (thy0, thy_text) =
  Data.extract_proofs thy0 thy_text
  |> map Json_Maker.json_proof
  |> String.concatWith " ISA_RL_SEP ";

(* write_json_proof: string -> (Data.env * Data.T Sections.T) -> unit *)
fun write_json_proof file_path proof = if Ops.file_exists file_path 
  then warning ("write_proof: File " ^ file_path ^ " already exists (skipping)")
  else Json_Maker.json_proof proof |> Print.write_to (Print.FILE file_path);

(* write_json_proofs: (string * theory * string) -> unit *)
fun write_json_proofs (write_dir, thy0, thy_text) =
  let 
    val proofs = Data.extract_proofs thy0 thy_text;
    fun fname j = ("proof" ^ Value.print_int j ^ ".json");
    fun write_next proof i =
      let
        val file_path = OS.Path.joinDirFile {dir=write_dir, file=fname i}
        val _ = write_json_proof file_path proof
      in i+1 end;
    val _ = fold write_next proofs 0
  in () end;

end;
