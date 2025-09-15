(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. A simple client reading until an end-of-prompt token.
*)

signature CLIENT =
sig
  val local_host: string
  val connect_to_server: int -> unit
  val communicate: string -> unit
  val get_last_response: unit -> string option
  val disconnect: unit -> unit
end;

structure Client: CLIENT =
struct

val end_of_prompt = "<|eop|>"
val local_host = "127.0.0.1"
val connection = Unsynchronized.ref (NONE : (BinIO.instream * BinIO.outstream) option)
val last_response = Unsynchronized.ref (NONE: string option)

fun connect_to_server port_num =
  let 
    val str_port = Value.print_int port_num;
    val lowest_port = 5000;
    val highest_port = 60000;
    val range_str = (Value.print_int lowest_port) ^ " and " ^ (Value.print_int highest_port)
  in if not (lowest_port < port_num andalso port_num < highest_port) then
    Output.error_message ("Invalid port=" ^ str_port ^ ". Choose between " ^ range_str ^ ".")
  else (case !connection of
       SOME _ => Output.error_message "Already connected.\n"
     | NONE =>
        let
          val server_str = local_host ^ ":" ^ (Value.print_int port_num);
          val streams = Socket_IO.open_streams server_str
          val _ = connection := SOME streams
          val _ = Output.writeln "Connected successfully.\n"
        in () end)
  end;

fun read_until stop_kwrd stream =
  let
    fun read_body bs = (case BinIO.input1 stream of
      NONE => if Bytes.is_empty bs then NONE else SOME (Bytes.content bs)
      | SOME b =>
        let
          val updated_buffer = Bytes.add (str (Byte.byteToChar b)) bs;
          val buffer_content = Bytes.content updated_buffer;
        in if Pred.contains stop_kwrd buffer_content 
            then SOME (Library.unsuffix stop_kwrd buffer_content)
          else read_body updated_buffer
        end)
  in read_body Bytes.empty end;

fun update_response in_stream =
  let
    val response = read_until end_of_prompt in_stream;
    val _ = (case response of 
      SOME str => Output.writeln ("Response is: " ^ str)
      | NONE => Output.writeln ("Response is NONE"))
  in last_response := response end;

fun send_message msg out_stream =
  let
    val full_message = Bytes.string (msg ^ end_of_prompt)
    val _ = Byte_Message.write out_stream [full_message];
  in Byte_Message.flush out_stream end;

fun communicate msg =
  (case !connection of
    SOME (in_stream, out_stream) => (send_message msg out_stream; update_response in_stream)
    | NONE => raise Fail "Not connected to server");

fun get_last_response () = !last_response;

fun disconnect () =
  (case !connection of
    SOME (in_stream, out_stream) =>
      (BinIO.closeIn in_stream;
       BinIO.closeOut out_stream;
       connection := NONE;
       Output.writeln "disconnected")
  | NONE => Output.writeln "already disconnected")

end;
