
signature WRITER =
sig
  (* G2TAC *)
  val write_g2tac_proof:  G2tac_Formatter.format -> string -> (Data.env * Data.T Sections.T) -> unit
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

(* write_g2tac_proof:  G2tac_Formatter.format -> string -> (Data.env * Data.T Sections.T) -> unit *)
fun write_g2tac_proof format write_file_path (env, proof) =
  let
    val steps = Sections.recover_trace proof
    fun write_next step prev =
      let
        val backslash = "\\";
        val start_quote = backslash ^ "<";
        val to_write = 
          G2tac_Formatter.pretty_step_of format prev env step
          |> Print.string_of_pretty
          |> Ops.replace_all start_quote (backslash ^ start_quote)
        val _ = Print.write_to (Print.FILE write_file_path) to_write
      in step end;
    val _ = fold write_next steps (hd steps)
  in () end;

(* write_g2tac_proof': (G2tac_Formatter.format * TextIO.outstream) list 
  -> Data.env * Data.T Sections.T 
  -> unit Exn.result list list *)
fun write_g2tac_proof' format_streams (env, proof) =
  let
    val steps = tl (Sections.recover_trace proof)
    fun write_next format_stream step (prev, errs) =
      let
        val format = fst format_stream
        val file_stream = snd format_stream
        val backslash = "\\";
        val start_quote = backslash ^ "<";
        val to_write = 
          G2tac_Formatter.pretty_step_of format prev env step
          |> Print.string_of_pretty
          |> Ops.replace_all start_quote (backslash ^ start_quote)
      in (step, Exn.result TextIO.output (file_stream, to_write) :: errs) end;
    fun write_all format_stream = 
      if null steps then 
        (Data.from (Data.s1_of env), [Exn.result (fn () => (hd steps; ())) ()])
      else fold (write_next format_stream) steps (hd steps, [])
    val errors = map (snd o write_all) format_streams
  in errors end;

(* write_g2tac_proofs: string * theory * string -> unit *)
fun write_g2tac_proofs (write_dir, thy0, thy_text) =
  let 
    val proofs = Data.extract_proofs thy0 thy_text;
    fun mk_file_streams_of format = 
      let 
        val file_name = "g2tac" ^ (G2tac_Formatter.str_number_of format) ^ ".txt"
        val file_path = OS.Path.joinDirFile {dir=write_dir, file=file_name}
        val file_stream = TextIO.openAppend file_path
      in (format, file_stream) end;
    val file_streams = map mk_file_streams_of G2tac_Formatter.formats_to_use;
    fun write_next proof errs =
      (write_g2tac_proof' file_streams proof) @ errs;
    val errors = fold write_next proofs [];
    val _ = map (TextIO.closeOut o snd) file_streams
    val _ = Par_Exn.release_all (maps rev errors)
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
