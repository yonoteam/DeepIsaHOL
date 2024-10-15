
signature WRITER =
sig
  val extract: theory * string -> string
  val write_proof: string -> (Data.env * Data.T Sections.T) -> unit
  val write_proofs: (string * theory * string) -> unit
end;


structure Writer: WRITER =
struct

(* val _ = ML_system_pp (fn _ => fn _ 
  => Pretty.to_polyml o pretty_json true 0 (Context.the_local_context ())) *)

(* extract: theory * string -> string *)
fun extract (thy0, thy_text) =
  Data.extract_proofs thy0 thy_text
  |> map Json_Maker.json_proof
  |> String.concatWith " ISA_RL_SEP ";

(* write_proof: string -> (Data.env * Data.T Sections.T) -> unit *)
fun write_proof file_path proof = if Ops.file_exists file_path 
  then warning ("write_proof: File " ^ file_path ^ " already exists (skipping)")
  else Json_Maker.json_proof proof |> Print.write_to (Print.FILE file_path);

(* write_proofs: (string * theory * string) -> unit *)
fun write_proofs (write_dir, thy0, thy_text) =
  let 
    val proofs = Data.extract_proofs thy0 thy_text;
    fun fname j = ("proof" ^ Value.print_int j ^ ".json");
    fun write_next proof i =
      let
        val file_path = OS.Path.joinDirFile {dir=write_dir, file=fname i}
        val _ = write_proof file_path proof
      in i+1 end;
    val _ = fold write_next proofs 0
  in () end;

end;
